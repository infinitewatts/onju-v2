"""Local deterministic tests: no audio, network, permissions, or live actions."""
import concurrent.futures
import copy
import os
from pathlib import Path
import tempfile
import unittest

from pipeline.orchestrator_bridge import BridgeError, BridgeStore, resolve_reply


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.now = 1000
        self.config = dict(mode='live', host_session_id='host-1', device='onju-1',
                           state_path=str(Path(self.tmp.name) / 'state' / 'bridge.sqlite'))
        self.store = BridgeStore(self.config, clock=lambda: self.now)
        self.identity = dict(host_session_id='host-1', session_id='chat-1', question_id='q-1', version=1)
        self.payload = dict(self.identity, op='submit', kind='structured', text='Choose a direction',
                            summary='Direction', options=[{'label': 'Left', 'description': 'Go left'},
                                                          {'label': 'Right'}],
                            requires_desktop=False, expires_at=1200)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def call(self, op, **kw):
        return self.store.dispatch(dict(self.identity, op=op, **kw))

    def submit(self, **kw):
        return self.store.dispatch(dict(self.payload, **kw))['question']

    def capture(self):
        self.submit()
        self.call('announced')
        return self.call('propose', answer_id='a-1', answer={'choice': 1, 'text': 'Left'})

    def test_lifecycle_requires_emission_and_ack(self):
        self.assertEqual(self.capture()['question']['status'], 'captured')
        with self.assertRaises(BridgeError):
            self.call('ack', answer_id='a-1')
        emitted = self.call('emitted')
        self.assertEqual(emitted['answer_id'], 'a-1')
        self.assertTrue(emitted['question']['unverified'])
        self.assertEqual(self.call('emitted'), emitted)
        with self.assertRaises(BridgeError):
            self.call('ack', answer_id='wrong')
        ack = self.call('ack', answer_id='a-1')
        self.assertEqual(ack['question']['status'], 'acknowledged')
        self.assertEqual(self.call('ack', answer_id='a-1'), ack)
        self.assertIsNone(self.call('next')['question'])

    def test_restart_preserves_proposal_and_replay(self):
        captured = self.capture()
        self.store.close()
        self.store = BridgeStore(self.config, clock=lambda: self.now)
        self.assertEqual(self.call('status'), captured)
        self.assertEqual(self.submit(), captured['question'])
        self.assertEqual(self.call('emitted')['answer_id'], 'a-1')

    def test_same_version_conflict_and_newer_supersession(self):
        self.submit()
        with self.assertRaises(BridgeError):
            self.submit(text='Different question')
        self.assertEqual(self.submit(version=2)['status'], 'queued')
        self.assertEqual(self.call('status')['question']['status'], 'superseded')
        with self.assertRaises(BridgeError):
            self.submit(version=1, text='Old changed question')
        with self.assertRaises(BridgeError):
            self.call('announced')

    def test_cannot_supersede_emitted_proposal(self):
        self.capture()
        self.call('emitted')
        with self.assertRaises(BridgeError):
            self.submit(version=2)

    def test_fifo_and_only_active_question_can_announce(self):
        self.submit()
        self.submit(question_id='q-2')
        self.assertEqual(self.call('next')['question']['question_id'], 'q-1')
        with self.assertRaises(BridgeError):
            self.call('announced', question_id='q-2')
        self.call('cancel')
        self.assertEqual(self.call('next')['question']['question_id'], 'q-2')

    def test_lease_expiry_rejects_late_reply(self):
        self.submit()
        self.call('announced')
        self.now += 10
        with self.assertRaises(BridgeError):
            self.call('propose', answer_id='a-1', answer={'choice': 1, 'text': 'Left'})
        self.assertEqual(self.call('status')['question']['status'], 'expired')
        self.assertIsNone(self.call('next')['question'])

    def test_renew_is_bounded_and_absolute_expiry_wins(self):
        self.submit(expires_at=1015)
        self.now += 7
        self.assertEqual(self.call('renew')['question']['lease_expires_at'], 1015)
        self.now = 1015
        with self.assertRaises(BridgeError):
            self.call('renew')
        self.assertEqual(self.call('status')['question']['status'], 'expired')

    def test_observe_is_terminal_and_does_not_expire_or_play(self):
        self.store.close()
        self.store = BridgeStore(dict(self.config, mode='observe'), clock=lambda: self.now)
        with self.assertLogs('pipeline.orchestrator_bridge', level='INFO') as logs:
            q = self.submit()
        self.assertIn('would_announce', logs.output[0])
        self.assertNotIn('Choose a direction', logs.output[0])
        self.assertEqual(q['status'], 'observed')
        self.assertIsNone(q['lease_expires_at'])
        self.now += 1000
        self.assertEqual(self.call('status')['question']['status'], 'observed')
        self.assertIsNone(self.call('next')['question'])
        with self.assertRaises(BridgeError):
            self.call('announced')

    def test_off_rejects_every_operation(self):
        self.store.close()
        self.store = BridgeStore(dict(self.config, mode='off'), clock=lambda: self.now)
        for op in ('submit', 'next', 'status', 'cancel'):
            with self.subTest(op=op), self.assertRaises(BridgeError):
                self.store.dispatch(dict(self.payload, op=op))

    def test_host_session_and_device_mixups_rejected_after_restart(self):
        self.submit()
        for changes in ({'host_session_id': 'other'}, {'session_id': 'other'}, {'device': 'other'}):
            with self.subTest(changes=changes), self.assertRaises(BridgeError):
                self.store.dispatch(dict(self.identity, op='announced', **changes))
        self.store.close()
        self.store = BridgeStore(dict(self.config, host_session_id='host-2'), clock=lambda: self.now)
        self.assertIsNone(self.store.dispatch({'op': 'next'})['question'])
        with self.assertRaises(BridgeError):
            self.store.dispatch(dict(self.identity, host_session_id='host-2', op='status'))

    def test_cancel_and_screen_before_and_after_emission(self):
        self.capture()
        self.assertEqual(self.call('screen')['question']['status'], 'needs_desktop')
        with self.assertRaises(BridgeError):
            self.call('emitted')
        self.identity['question_id'] = self.payload['question_id'] = 'q-2'
        self.capture()
        self.call('emitted')
        for op in ('screen', 'cancel'):
            result = self.call(op)
            self.assertTrue(result['correction_required'])
            self.assertEqual(result['question']['status'], 'emitted')

    def test_desktop_question_cannot_capture(self):
        self.submit(requires_desktop=True)
        self.call('announced')
        with self.assertRaises(BridgeError):
            self.call('propose', answer_id='a-1', answer={'choice': 1, 'text': 'Left'})

    def test_strict_submit_validation(self):
        cases = [dict(version=True), dict(text=' '), dict(text='x' * 2001),
                 dict(options='bad'), dict(options=[{'label': 'x' * 121}]),
                 dict(options=[{'label': 'x'}] * 9), dict(requires_desktop=1),
                 dict(expires_at=True), dict(expires_at=float('nan')),
                 dict(expires_at=1601), dict(expires_at=1000), dict(summary='x' * 501),
                 dict(session_id='../x'), dict(tool_use_id=3), dict(ambient='secret')]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(BridgeError):
                self.submit(**changes)
        self.assertIsNone(self.call('next')['question'])

    def test_pending_capacity_has_no_silent_eviction(self):
        for number in range(32):
            self.submit(question_id='q-' + str(number))
        with self.assertRaises(BridgeError):
            self.submit(question_id='overflow')
        self.assertEqual(self.call('next')['question']['question_id'], 'q-0')
        self.submit(question_id='q-0', version=2)
        self.assertEqual(self.call('next')['question']['question_id'], 'q-1')

    def test_total_capacity_includes_tombstones(self):
        self.store.close()
        self.store = BridgeStore(dict(self.config, mode='observe'), clock=lambda: self.now)
        for number in range(1024):
            self.submit(question_id='q-' + str(number))
        with self.assertRaises(BridgeError):
            self.submit(question_id='overflow')
        self.assertEqual(self.call('status')['question']['status'], 'observed')

    def test_cross_connection_atomic_race_has_one_winner(self):
        self.submit()
        self.call('announced')
        other = BridgeStore(self.config, clock=lambda: self.now)
        def propose(pair):
            store, answer_id = pair
            try:
                return store.dispatch(dict(self.identity, op='propose', answer_id=answer_id,
                                           answer={'choice': 1, 'text': 'Left'}))
            except BridgeError as exc:
                return exc.status
        try:
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                results = list(pool.map(propose, [(self.store, 'a-1'), (other, 'a-2')]))
            self.assertEqual(sum(type(result) is dict for result in results), 1)
            self.assertIn(409, results)
        finally:
            other.close()

    def test_slow_ack_survives_voice_lease_and_submit_expiry(self):
        self.capture()
        emitted = self.call('emitted')['question']
        self.assertEqual(emitted['receipt_expires_at'], 1600)
        self.now = 1250
        renewed = self.call('renew')['question']
        self.assertEqual(renewed['receipt_expires_at'], 1600)
        self.assertEqual(self.call('emitted')['question']['receipt_expires_at'], 1600)
        self.assertEqual(self.call('ack', answer_id='a-1')['question']['status'], 'acknowledged')

    def test_receipt_deadline_expires_without_indefinite_renewal(self):
        self.capture()
        self.call('emitted')
        self.now = 1599
        self.call('renew')
        self.now = 1600
        with self.assertRaises(BridgeError):
            self.call('ack', answer_id='a-1')
        self.assertEqual(self.call('status')['question']['status'], 'expired')
        self.assertEqual(self.call('next')['pending_count'], 0)
        self.assertTrue(self.call('cancel')['correction_required'])

    def test_cancel_context_restores_unique_answer_after_restart(self):
        self.capture()
        self.call('emitted')
        self.store.close()
        self.store = BridgeStore(self.config, clock=lambda: self.now)
        context = self.call('cancel_context')
        self.assertFalse(context['ambiguous'])
        self.assertEqual(context['question']['answer_id'], 'a-1')
        self.assertEqual(context['pending_count'], 1)
        self.assertNotIn('text', context['candidates'][0])
        self.assertTrue(self.call('cancel')['correction_required'])
        self.call('ack', answer_id='a-1')
        self.assertEqual(self.call('cancel_context')['question']['status'], 'acknowledged')
        self.now = 1600
        self.assertIsNone(self.call('cancel_context')['question'])

    def test_cancel_context_ambiguity_and_announced_priority(self):
        self.capture()
        self.submit(question_id='q-2', session_id='chat-2')
        second = dict(self.identity, question_id='q-2', session_id='chat-2')
        self.store.dispatch(dict(second, op='announced'))
        context = self.call('cancel_context')
        self.assertEqual(context['question']['question_id'], 'q-2')
        self.assertFalse(context['ambiguous'])
        self.store.dispatch(dict(second, op='propose', answer_id='a-2',
                                 answer={'choice': 2, 'text': 'Right'}))
        self.store.close()
        self.store = BridgeStore(self.config, clock=lambda: self.now)
        context = self.call('cancel_context')
        self.assertTrue(context['ambiguous'])
        self.assertIsNone(context['question'])
        self.assertEqual(len(context['candidates']), 2)
        self.assertEqual(context['pending_count'], 2)
        for candidate in context['candidates']:
            self.assertNotIn('text', candidate)
        self.assertEqual(self.call('status')['question']['status'], 'captured')

    def test_multiple_questions_capture_before_any_emission(self):
        self.submit()
        self.submit(question_id='q-2')
        self.call('announced')
        self.call('propose', answer_id='a-1', answer={'choice': 1, 'text': 'Left'})
        self.assertEqual(self.call('next')['question']['question_id'], 'q-2')
        self.call('announced', question_id='q-2')
        self.call('renew')
        self.call('propose', question_id='q-2', answer_id='a-2', answer={'choice': 2, 'text': 'Right'})
        self.assertIsNone(self.call('next')['question'])
        self.assertEqual(self.call('emitted')['answer_id'], 'a-1')
        self.assertEqual(self.call('emitted', question_id='q-2')['answer_id'], 'a-2')

    def test_receipt_claim_is_durable_and_only_after_ack(self):
        self.capture()
        self.assertIsNone(self.call('next_receipt')['question'])
        with self.assertRaises(BridgeError):
            self.call('receipt_spoken', answer_id='a-1')
        self.call('emitted')
        self.call('ack', answer_id='a-1')
        self.assertEqual(self.call('next_receipt')['question']['question_id'], 'q-1')
        self.assertTrue(self.call('receipt_spoken', answer_id='a-1')['claimed'])
        self.assertFalse(self.call('receipt_spoken', answer_id='a-1')['claimed'])
        self.store.close()
        self.store = BridgeStore(self.config, clock=lambda: self.now)
        self.assertIsNone(self.call('next_receipt')['question'])

    def test_private_state_permissions(self):
        state = Path(self.config['state_path'])
        self.assertEqual(os.stat(state).st_mode & 0o777, 0o600)
        self.assertEqual(os.stat(state.parent).st_mode & 0o777, 0o700)


class ReplyTests(unittest.TestCase):
    question = {'options': [{'label': 'Red'}, {'label': 'Green'}, {'label': 'Blue'}]}

    def parse(self, text):
        return resolve_reply(text, self.question)

    def test_exact_choices_and_explicit_correction(self):
        for text in ('three', '3', 'option three', 'Blue', 'two actually three', 'two, I mean three'):
            self.assertEqual(self.parse(text), {'intent': 'choice', 'answer': {'choice': 3, 'text': 'Blue'}})
        for text in ('two three', 'two or three', 'two and three', 'Red or Green'):
            self.assertEqual(self.parse(text)['intent'], 'clarify')

    def test_explicit_freeform_preserves_original_wording(self):
        for prefix in ('my answer is ', 'Tell Orchestrator '):
            self.assertEqual(self.parse(prefix + 'Keep My CAPS, please.'),
                             {'intent': 'freeform', 'answer': {'choice': None, 'text': 'Keep My CAPS, please.'}})

    def test_unrelated_commands_never_become_answers(self):
        for text in ('turn on the lights', 'play option two tomorrow', 'call Eric', 'greenish', 'turn on lights actually three', ''):
            self.assertEqual(self.parse(text), {'intent': 'unrelated'})

    def test_commands_and_details(self):
        for text, intent in [('repeat that', 'repeat'), ('skip', 'skip'),
                             ("what's pending", 'pending'), ('show it on screen', 'screen'),
                             ('cancel that answer', 'cancel')]:
            self.assertEqual(self.parse(text), {'intent': intent})
        self.assertEqual(self.parse('explain option two'), {'intent': 'details', 'choice': 2})
        self.assertEqual(self.parse('explain option eight'), {'intent': 'clarify'})

    def test_duplicate_labels_clarify(self):
        self.assertEqual(resolve_reply('Green', {'options': [{'label': 'Green'}, {'label': 'Green'}]}),
                         {'intent': 'clarify'})


if __name__ == '__main__':
    unittest.main()
