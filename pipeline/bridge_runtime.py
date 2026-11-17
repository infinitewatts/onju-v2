"""Optional local speaker transport; voice captures are unverified proposals."""
import asyncio
from collections import OrderedDict
import logging
import time
import uuid

from pipeline.audio import opus_encode, opus_frames_to_tcp_payload
from pipeline.orchestrator_bridge import BridgeError, BridgeStore, resolve_reply
from pipeline.protocol import send_audio, close_led_connection
from pipeline.security import validate_device
from pipeline.services import tts

log = logging.getLogger(__name__)
KEYS = ('host_session_id', 'session_id', 'question_id', 'version')


def identity(question):
    return {key: question[key] for key in KEYS}


class BridgeRuntime:
    def __init__(self, config, manager, store=None):
        self.config, self.manager = config, manager
        settings = config.get('orchestrator_bridge', {})
        self.mode = settings.get('mode', 'off')
        self.store = store if store is not None else (BridgeStore(settings) if self.mode != 'off' else None)
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.closed = False
        self.task = None

    @property
    def enabled(self):
        return self.mode != 'off' and not self.closed

    def dispatch(self, payload):
        if not self.enabled:
            raise BridgeError(404, 'Bridge is unavailable')
        if isinstance(payload, dict) and payload.get('op') == 'health':
            if set(payload) != {'op'}:
                raise BridgeError(400, 'Invalid health request')
            return {'mode': self.mode, 'host_session_id': self.store.host_session_id,
                    'device': self.store.device, 'closed': self.closed}
        return self.store.dispatch(payload)

    def capture_ref(self, device):
        if self.mode != 'live' or self.closed or device.hostname != self.store.device:
            return None
        q = self.dispatch({'op': 'next'})['question']
        if q and q['status'] == 'announced':
            return identity(q)
        context = self.dispatch({'op': 'cancel_context'})
        if context['ambiguous']:
            return {'ambiguous': True}
        return identity(context['question']) if context['question'] else None

    def start(self):
        if self.mode == 'live' and self.task is None:
            self.task = asyncio.create_task(self.run())

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        self.cache.clear()
        self.cache_bytes = 0
        if self.store:
            self.store.close()

    def _current(self, ref, statuses):
        try:
            q = self.dispatch(dict(ref, op='status'))['question']
            return q if q['status'] in statuses else None
        except BridgeError:
            return None

    @staticmethod
    def question_line(q):
        if q['requires_desktop']:
            return 'Orchestrator needs your attention on screen. Please answer on your computer.'
        options = ' '.join(f"Option {i}: {o['label']}." for i, o in enumerate(q['options'], 1))
        return f"Orchestrator asks: {q.get('summary') or q['text']} {options} Say Hey Aiden, then your option, or my answer is, followed by your answer."

    def _device(self):
        d = self.manager.devices.get(self.store.device)
        if d is None or d.ptt or getattr(d, 'wake_gate', None) is None:
            return None
        validate_device(self.config, d.hostname, d.ip)
        return d

    @staticmethod
    def _idle(d):
        return not (d.processing or getattr(d, 'queued_utterances', 0)
                    or getattr(d, 'pilot_speaking', False) or d.ptt_buffer
                    or d.vad.recording or d.wake_gate.active)

    @staticmethod
    def _reset(d):
        d.wake_gate.reset()
        d.vad.reset()
        d.vad.pre_buffer.clear()

    async def speak(self, d, line, valid=lambda: True):
        key = (d.voice, line)
        pcm = self.cache.get(key)
        if pcm is None:
            pcm = await tts.synthesize(line, d.voice, self.config)
            size = len(pcm)
            if size <= 2_000_000:
                while self.cache and (len(self.cache) >= 8 or self.cache_bytes + size > 2_000_000):
                    _, old = self.cache.popitem(last=False)
                    self.cache_bytes -= len(old)
                self.cache[key] = pcm
                self.cache_bytes += size
        else:
            self.cache.move_to_end(key)
        if self.closed or not valid():
            return False
        frames = opus_encode(pcm, self.config['audio']['sample_rate'], self.config['audio']['opus_frame_size'])
        if not valid():
            return False
        cfg = self.config['device']
        sent = await send_audio(d.ip, self.config['network']['tcp_port'], opus_frames_to_tcp_payload(frames),
                                mic_timeout=cfg['default_mic_timeout'], volume=cfg['default_volume'],
                                fade=cfg['led_fade'], may_send=lambda: not self.closed and valid())
        return sent is not False

    async def tick(self):
        if self.mode != 'live' or self.closed:
            return
        d = self._device()
        if d is None or not self._idle(d):
            return
        receipt = self.dispatch({'op': 'next_receipt'})['question']
        q = receipt or self.dispatch({'op': 'next'})['question']
        if not q or (not receipt and q['status'] != 'queued'):
            return
        # No await between idle check and reservation. The sole run task owns playback.
        d.processing = True
        d.bridge_speaking = True
        ref = identity(q)
        started = time.monotonic()
        try:
            self._reset(d)
            if d.vad_writer is not None:
                await close_led_connection(d.vad_writer)
                d.vad_writer = None
            if receipt:
                # Durable at-most-once attempt, including process restart; not evidence of hearing.
                claimed = self.dispatch(dict(ref, op='receipt_spoken', answer_id=q['answer_id']))['claimed']
                if claimed:
                    await self.speak(d, 'Orchestrator received your answer.',
                                     lambda: bool(self._current(ref, {'acknowledged'})))
            else:
                sent = await self.speak(d, self.question_line(q),
                                        lambda: bool(self._current(ref, {'queued'})))
                if sent and self._current(ref, {'queued'}):
                    self.dispatch(dict(ref, op='screen' if q['requires_desktop'] else 'announced'))
        finally:
            try:
                self._reset(d)
            finally:
                d.bridge_speaking = False
                d.processing = False
                log.info('orchestrator_bridge playback_elapsed_s=%.3f', time.monotonic() - started)

    async def run(self):
        while not self.closed:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning('Bridge transport failed (%s)', type(exc).__name__)
                await asyncio.sleep(1)
            await asyncio.sleep(.2)

    async def intercept(self, device, text, ref):
        if self.mode != 'live' or self.closed or device.hostname != self.store.device:
            return False
        q = self._current(ref, {'announced', 'captured', 'emitted', 'acknowledged'}) if ref else None
        parsed = resolve_reply(text, q or {})
        intent = parsed['intent']
        if intent == 'unrelated':
            return False
        line = 'That question is no longer accepting an answer. Please check your computer.'
        if intent == 'pending':
            pending = self.dispatch({'op': 'next'})['pending_count']
            line = f'There are {pending} pending Orchestrator questions.' if pending else 'There are no pending Orchestrator questions.'
        elif ref and ref.get('ambiguous'):
            line = 'More than one answer could match. Please correct it on your computer.'
        elif q:
            if intent in ('choice', 'freeform'):
                if q['requires_desktop']:
                    line = 'Please answer this question on your computer.'
                elif q['status'] == 'announced':
                    try:
                        self.dispatch(dict(ref, op='propose', answer_id=uuid.uuid4().hex, answer=parsed['answer']))
                        line = 'Answer captured. Waiting for Orchestrator to receive it.'
                    except BridgeError:
                        pass
                else:
                    line = 'Your answer is already captured. Please check your computer for changes.'
            elif intent in ('screen', 'skip', 'cancel'):
                result = self.dispatch(dict(ref, op='cancel' if intent == 'cancel' else 'screen'))
                line = ('That answer was already emitted. Please correct it on your computer.' if result.get('correction_required')
                        else 'Please continue on your computer.' if intent != 'cancel' else 'Answer cancelled.')
            elif intent == 'repeat':
                line = self.question_line(q)
            elif intent == 'details':
                option = q['options'][parsed['choice'] - 1]
                line = option.get('description') or option['label']
            elif intent == 'clarify':
                line = 'Please say one option number, or say my answer is, followed by your answer.'
        self._reset(device)
        valid = lambda: True
        if q and intent in ('repeat', 'details', 'clarify'):
            valid = lambda: bool(self._current(ref, {'announced'}))
        elif q and intent in ('choice', 'freeform'):
            valid = lambda: bool(self._current(ref, {'captured', 'emitted', 'acknowledged'}))
        await self.speak(device, line, valid)
        self._reset(device)
        return True
