"""Bounded local Orchestrator question queue. Voice proposals are never authority."""
import json
import logging
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time

LOG = logging.getLogger(__name__)
TERMINAL = frozenset({'observed', 'acknowledged', 'cancelled', 'expired',
                      'superseded', 'needs_desktop'})
PENDING = frozenset({'queued', 'announced', 'captured', 'emitted'})
ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z')


class BridgeError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _fail(message, status=400):
    raise BridgeError(status, message)


def _string(value, name, maximum, nonempty=True):
    if type(value) is not str or len(value) > maximum or (nonempty and not value.strip()):
        _fail('Invalid ' + name)
    return value


def _id(value, name):
    if type(value) is not str or not ID.fullmatch(value):
        _fail('Invalid ' + name)
    return value


def _version(value):
    if type(value) is not int or not 1 <= value <= 2147483647:
        _fail('Invalid version')
    return value


class BridgeStore:
    """SQLite transactions serialize transitions across threads and processes."""
    def __init__(self, config, clock=None):
        self.mode = config.get('mode', 'off')
        if self.mode not in ('off', 'observe', 'live'):
            _fail('Invalid bridge mode')
        self.host_session_id = _id(config.get('host_session_id'), 'host_session_id')
        self.device = _id(config.get('device'), 'device')
        self.clock = clock or time.time
        self.lock = threading.RLock()
        path = Path(_string(config.get('state_path'), 'state_path', 4096)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.parent.is_symlink() or path.is_symlink():
            _fail('State path must not be a symlink')
        os.chmod(path.parent, 0o700)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(path, 0o600)
        self.db = sqlite3.connect(str(path), timeout=10, check_same_thread=False,
                                  isolation_level=None)
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('CREATE TABLE IF NOT EXISTS questions '
                        '(seq INTEGER PRIMARY KEY AUTOINCREMENT, '
                        'session TEXT NOT NULL, qid TEXT NOT NULL, version INTEGER NOT NULL, '
                        'payload TEXT NOT NULL, snapshot TEXT NOT NULL, '
                        'UNIQUE(session,qid,version))')

    def close(self):
        with self.lock:
            self.db.close()

    def _rows(self):
        return [(seq, json.loads(body)) for seq, body in self.db.execute(
            'SELECT seq,snapshot FROM questions ORDER BY seq')]

    def _save(self, seq, q):
        self.db.execute('UPDATE questions SET snapshot=? WHERE seq=?',
                        (json.dumps(q, sort_keys=True), seq))

    def _expire(self, now):
        for seq, q in self._rows():
            voice_expired = (q['status'] in ('queued', 'announced', 'captured')
                             and (q['expires_at'] <= now or q['lease_expires_at'] <= now))
            receipt_expired = (q['status'] == 'emitted'
                               and q.get('receipt_expires_at', q['expires_at']) <= now)
            if voice_expired or receipt_expired:
                q['status'] = 'expired'
                self._save(seq, q)

    def _active(self):
        return next(((seq, q) for seq, q in self._rows()
                     if q['host_session_id'] == self.host_session_id
                     and q['device'] == self.device and q['mode'] == self.mode
                     and q['status'] in ('queued', 'announced')), None)

    def _identity(self, p):
        for name in ('host_session_id', 'session_id', 'question_id'):
            _id(p.get(name), name)
        _version(p.get('version'))
        if p['host_session_id'] != self.host_session_id:
            _fail('Host session mismatch', 409)
        if 'device' in p and p['device'] != self.device:
            _fail('Device mismatch', 409)

    def _find(self, p):
        row = self.db.execute('SELECT seq,snapshot FROM questions WHERE '
                              'session=? AND qid=? AND version=?',
                              (p['session_id'], p['question_id'], p['version'])).fetchone()
        if not row:
            _fail('Unknown question', 404)
        q = json.loads(row[1])
        if q['host_session_id'] != self.host_session_id or q['device'] != self.device or q['mode'] != self.mode:
            _fail('Question belongs to another host, device or mode', 409)
        return row[0], q

    def _submit(self, p, now):
        self._identity(p)
        if p.get('kind') not in ('structured', 'explicit'):
            _fail('Invalid question kind')
        _string(p.get('text'), 'text', 2000)
        if 'summary' in p:
            _string(p['summary'], 'summary', 500)
        if 'tool_use_id' in p and p['tool_use_id'] is not None:
            _id(p['tool_use_id'], 'tool_use_id')
        options = p.get('options', [])
        if type(options) is not list or len(options) > 8:
            _fail('Invalid options')
        for option in options:
            if type(option) is not dict or set(option) - {'label', 'description'}:
                _fail('Invalid option')
            _string(option.get('label'), 'option label', 120)
            if 'description' in option:
                _string(option['description'], 'option description', 500)
        if type(p.get('requires_desktop')) is not bool:
            _fail('Invalid requires_desktop')
        expires = p.get('expires_at')
        if type(expires) not in (int, float) or not math.isfinite(expires):
            _fail('Invalid expires_at')
        allowed = {'op', 'host_session_id', 'session_id', 'question_id', 'version',
                   'device', 'kind', 'tool_use_id', 'text', 'summary', 'options',
                   'expires_at', 'requires_desktop'}
        if set(p) - allowed:
            _fail('Unknown submit fields')
        canonical = {k: v for k, v in p.items() if k != 'op'}
        encoded = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
        existing = self.db.execute('SELECT payload,snapshot FROM questions WHERE '
                                   'session=? AND qid=? AND version=?',
                                   (p['session_id'], p['question_id'], p['version'])).fetchone()
        if existing:
            old = json.loads(existing[1])
            if old['host_session_id'] != self.host_session_id or existing[0] != encoded:
                _fail('Conflicting question version', 409)
            return {'question': old}
        if not now < expires <= now + 600:
            _fail('Question expiry must be within 600 seconds')
        previous = [(seq, q) for seq, q in self._rows()
                    if q['session_id'] == p['session_id'] and q['question_id'] == p['question_id']]
        if previous and max(q['version'] for _, q in previous) >= p['version']:
            _fail('Stale question version', 409)
        rows = self._rows()
        if len(rows) >= 1024:
            _fail('Question storage capacity reached', 409)
        retiring = {seq for seq, q in previous if q['status'] in PENDING}
        if self.mode == 'live' and sum(q['status'] in PENDING and seq not in retiring for seq, q in rows) >= 32:
            _fail('Pending question capacity reached', 409)
        for seq, q in previous:
            if q['status'] in PENDING:
                if q['status'] == 'emitted':
                    _fail('Emitted answer requires acknowledgement before replacement', 409)
                q['status'] = 'superseded'
                self._save(seq, q)
        q = dict(canonical, mode=self.mode, device=self.device, options=options,
                 status='observed' if self.mode == 'observe' else 'queued',
                 lease_expires_at=None if self.mode == 'observe' else min(expires, now + 10))
        self.db.execute('INSERT INTO questions(session,qid,version,payload,snapshot) VALUES(?,?,?,?,?)',
                        (p['session_id'], p['question_id'], p['version'], encoded,
                         json.dumps(q, sort_keys=True)))
        return {'question': q}

    def dispatch(self, payload):
        if type(payload) is not dict or type(payload.get('op')) is not str:
            _fail('Operation required')
        op = payload['op']
        if op not in {'submit', 'status', 'next', 'renew', 'announced', 'propose', 'emitted', 'ack', 'cancel', 'screen', 'next_receipt', 'receipt_spoken', 'cancel_context'}:
            _fail('Unknown operation')
        if self.mode == 'off':
            _fail('Bridge is off', 409)
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                now = self.clock()
                self._expire(now)
                result = self._dispatch(op, payload, now)
                self.db.execute('COMMIT')
            except Exception:
                # Expiry is also applied at the next successful read/transition.
                self.db.execute('ROLLBACK')
                raise
        q = result.get('question')
        LOG.info('orchestrator_bridge op=%s mode=%s question_id=%s version=%s status=%s',
                 'would_announce' if op == 'submit' and self.mode == 'observe' else op,
                 self.mode, q.get('question_id') if q else None,
                 q.get('version') if q else None, q.get('status') if q else None)
        return result

    def _dispatch(self, op, p, now):
        if op == 'submit':
            return self._submit(p, now)
        if op in ('next', 'next_receipt', 'cancel_context'):
            if 'host_session_id' in p and p['host_session_id'] != self.host_session_id:
                _fail('Host session mismatch', 409)
        scoped = [q for _, q in self._rows()
                  if q['host_session_id'] == self.host_session_id
                  and q['device'] == self.device and q['mode'] == self.mode]
        pending_count = sum(q['status'] in PENDING for q in scoped)
        if op == 'cancel_context':
            announced = next((q for q in scoped if q['status'] == 'announced'), None)
            candidates = [q for q in scoped
                          if q['status'] in ('captured', 'emitted')
                          or (q['status'] == 'acknowledged' and
                              q.get('receipt_expires_at', 0) > now)]
            metadata = [{key: q[key] for key in
                         ('host_session_id', 'session_id', 'question_id', 'version',
                          'device', 'answer_id', 'status')} for q in candidates]
            question = announced or (candidates[0] if len(candidates) == 1 else None)
            return {'question': question, 'ambiguous': announced is None and len(candidates) > 1,
                    'candidates': metadata, 'pending_count': pending_count}
        if op == 'next_receipt':
            if self.mode != 'live':
                return {'question': None}
            receipt = next((q for _, q in self._rows()
                            if q['host_session_id'] == self.host_session_id
                            and q['device'] == self.device and q['mode'] == self.mode
                            and q['status'] == 'acknowledged' and not q.get('ack_spoken')), None)
            return {'question': receipt}
        if op == 'next':
            if 'host_session_id' in p and p['host_session_id'] != self.host_session_id:
                _fail('Host session mismatch', 409)
            active = self._active()
            return {'question': active[1] if active else None, 'pending_count': pending_count}
        self._identity(p)
        seq, q = self._find(p)
        if op == 'status':
            return {'question': q}
        if op == 'receipt_spoken':
            _id(p.get('answer_id'), 'answer_id')
            if p['answer_id'] != q.get('answer_id'):
                _fail('Receipt answer mismatch', 409)
            if self.mode != 'live' or q['status'] != 'acknowledged':
                _fail('No acknowledged receipt', 409)
            claimed = not q.get('ack_spoken', False)
            q['ack_spoken'] = True
            self._save(seq, q)
            return {'question': q, 'claimed': claimed}
        if 'reason' in p:
            _string(p['reason'], 'reason', 120)
        if op == 'ack' and q['status'] == 'acknowledged':
            if p.get('answer_id') != q.get('answer_id'):
                _fail('Answer acknowledgement mismatch', 409)
            return {'question': q}
        if self.mode != 'live':
            _fail('Observe mode cannot change question state', 409)
        if op in ('cancel', 'screen'):
            if q['status'] in ('emitted', 'acknowledged') or 'receipt_expires_at' in q:
                return {'question': q, 'correction_required': True}
            target = 'cancelled' if op == 'cancel' else 'needs_desktop'
            if q['status'] == target:
                return {'question': q}
            if q['status'] not in PENDING:
                _fail('Question is closed', 409)
            q['status'] = target
        else:
            if q['status'] not in PENDING:
                _fail('Question is closed', 409)
            if (op == 'propose' and q['status'] == 'captured'
                    and p.get('answer_id') == q.get('answer_id')
                    and p.get('answer') == q.get('answer')):
                return {'question': q}
            active = self._active()
            if op in ('announced', 'propose') and (not active or active[0] != seq):
                _fail('Question is not active', 409)
            if op == 'renew':
                if q['status'] != 'emitted':
                    q['lease_expires_at'] = min(q['expires_at'], now + 10)
            elif op == 'announced':
                if q['status'] not in ('queued', 'announced'):
                    _fail('Question is no longer awaiting announcement', 409)
                q['status'] = 'announced'
            elif op == 'propose':
                _id(p.get('answer_id'), 'answer_id')
                answer = p.get('answer')
                if type(answer) is not dict or set(answer) != {'choice', 'text'}:
                    _fail('Invalid answer')
                _string(answer['text'], 'answer text', 2000)
                choice = answer['choice']
                if choice is not None and (type(choice) is not int or not 1 <= choice <= len(q['options'])):
                    _fail('Invalid answer choice')
                if q['requires_desktop']:
                    _fail('Question requires desktop', 409)
                if q['status'] == 'captured' and q.get('answer_id') == p['answer_id'] and q.get('answer') == answer:
                    return {'question': q}
                if q['status'] != 'announced':
                    _fail('Question is not awaiting an answer', 409)
                q.update(status='captured', answer_id=p['answer_id'], answer=answer, unverified=True)
            elif op == 'emitted':
                if q['status'] not in ('captured', 'emitted'):
                    _fail('No captured answer', 409)
                if q['status'] == 'captured':
                    q['receipt_expires_at'] = now + 600
                q['status'] = 'emitted'
            elif op == 'ack':
                _id(p.get('answer_id'), 'answer_id')
                if q['status'] != 'emitted' or q.get('answer_id') != p['answer_id']:
                    _fail('Answer acknowledgement mismatch', 409)
                q['status'] = 'acknowledged'
        self._save(seq, q)
        result = {'question': q}
        if op == 'emitted':
            result.update(answer_id=q['answer_id'], answer=q['answer'])
        return result


_NUMBERS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8}


def resolve_reply(text, question):
    """Parse deliberate local replies only; no fuzzy or ambient-command capture."""
    if type(text) is not str or not text.strip():
        return {'intent': 'unrelated'}
    original = text.strip()
    normalized = original.casefold().strip(' .!?')
    commands = {
        'repeat': 'repeat', 'repeat that': 'repeat', 'say that again': 'repeat',
        'skip': 'skip', 'skip this': 'skip', "what's pending": 'pending',
        'what is pending': 'pending', 'whats pending': 'pending',
        'on screen': 'screen', 'show on screen': 'screen', 'show it on screen': 'screen',
        'cancel that answer': 'cancel', 'cancel my answer': 'cancel',
    }
    if normalized in commands:
        return {'intent': commands[normalized]}
    prefix = re.match(r'^(?:my answer is|tell orchestrator)\s+(.+)$', original, re.I | re.S)
    if prefix:
        answer_text = prefix.group(1)
        return {'intent': 'freeform', 'answer': {'choice': None, 'text': answer_text}}
    options = question.get('options', [])
    details = re.fullmatch(r'(?:explain|details (?:on|for)) option (\d+|one|two|three|four|five|six|seven|eight)', normalized)
    if details:
        token = details.group(1)
        choice = _NUMBERS.get(token, int(token) if token.isdigit() else 0)
        return {'intent': 'details', 'choice': choice} if 1 <= choice <= len(options) else {'intent': 'clarify'}
    # Only explicit correction markers discard an earlier choice.
    parts = re.split(r'\b(?:actually|i mean|rather)\b', normalized)
    if len(parts) > 1:
        earlier = parts[0].strip(' ,')
        if resolve_reply(earlier, question).get('intent') != 'choice':
            return {'intent': 'unrelated'}
    corrected = parts[-1].strip(' ,')
    candidate = re.sub(r'^(?:option|choice|number)\s+', '', corrected)
    value = _NUMBERS.get(candidate, int(candidate) if candidate.isdigit() and len(candidate) < 4 else None)
    if value is None:
        matches = [i for i, option in enumerate(options, 1)
                   if candidate == option['label'].casefold().strip(' .!?')]
        if len(matches) == 1:
            value = matches[0]
        elif len(matches) > 1:
            return {'intent': 'clarify'}
    if value is not None:
        if 1 <= value <= len(options):
            return {'intent': 'choice', 'answer': {'choice': value, 'text': options[value - 1]['label']}}
        return {'intent': 'clarify'}
    tokens = re.findall(r'\b(?:[1-8]|one|two|three|four|five|six|seven|eight)\b', candidate)
    labels = [option['label'].casefold().strip(' .!?') for option in options]
    mentioned_labels = [label for label in labels if label and
                        re.search(r'(?<!\w)' + re.escape(label) + r'(?!\w)', candidate)]
    if len(tokens) > 1 or len(mentioned_labels) > 1:
        return {'intent': 'clarify'}
    return {'intent': 'unrelated'}
