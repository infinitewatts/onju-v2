"""Gate a real child process to verify progressive output and teardown."""
import asyncio
import json
import os
import stat
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import uvicorn
from starlette.requests import Request

from pipeline.services import voice_agent

FAKE = r'''#!PYTHON
import json, os, sys, time
from pathlib import Path
root = Path(os.environ['STREAM_TEST_ROOT'])
(root / 'pid').write_text(str(os.getpid()))
(root / 'args').write_text(json.dumps(sys.argv[1:]))
(root / 'input').write_text(sys.stdin.read())
mode = os.environ.get('STREAM_TEST_MODE', 'gated')
def emit(obj):
    print(json.dumps(obj), flush=True)
def delta(kind, **value):
    emit({'type': 'stream_event', 'event': {'type': 'content_block_delta',
          'delta': {'type': kind, **value}}})
if mode == 'stderr':
    sys.stderr.write('private stderr ' * 200000)
    sys.stderr.flush()
delta('thinking_delta', thinking='SECRET THINKING')
delta('input_json_delta', partial_json='SECRET TOOL JSON')
emit({'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': 'DUPLICATE'}]}})
delta('text_delta', text='Hello. ')
if mode in ('gated', 'blocked'):
    while not (root / 'gate').exists():
        time.sleep(.01)
if mode == 'malformed':
    print('not json', flush=True)
    sys.exit(0)
if mode == 'noresult':
    sys.exit(0)
if mode == 'failure':
    emit({'type': 'result', 'is_error': True, 'result': 'internal failure'})
    sys.exit(1)
delta('text_delta', text='Ready.')
emit({'type': 'result', 'is_error': False, 'result': 'Hello. Ready.'})
(root / 'finished').write_text('yes')
'''


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        child = self.root / 'claude'
        child.write_text(FAKE.replace('PYTHON', sys.executable, 1))
        child.chmod(child.stat().st_mode | stat.S_IEXEC)
        args = voice_agent.parse_args(['--workdir', str(self.root / 'state'),
                                       '--claude-bin', str(child)])
        self.settings = voice_agent.Settings(args)
        self.settings.workdir.mkdir()
        self.store = voice_agent.SessionStore(self.settings.state_file, 1800)
        self.lock = asyncio.Lock()
        self.env = patch.dict(os.environ, {'STREAM_TEST_ROOT': str(self.root),
                                          'STREAM_TEST_MODE': 'gated',
                                          'VOICE_AGENT_TOKEN': ''})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def turn(self):
        return voice_agent.stream_turn(self.settings, self.store, self.lock,
                                       'onju-8DAB80', 'User speech on stdin.')

    def assert_reaped(self):
        pid = int((self.root / 'pid').read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertFalse(self.lock.locked())

    async def test_delta_arrives_before_child_is_allowed_to_finish(self):
        turn = self.turn()
        try:
            self.assertEqual(await asyncio.wait_for(anext(turn), 3), 'Hello. ')
            self.assertFalse((self.root / 'finished').exists())
            self.assertTrue(self.lock.locked())
            self.assertEqual(self.store.data, {})  # not a successful turn yet
            args = json.loads((self.root / 'args').read_text())
            self.assertEqual(args[args.index('--output-format') + 1], 'stream-json')
            self.assertIn('--verbose', args)
            self.assertIn('--include-partial-messages', args)
            self.assertIn('--bare', args)
            self.assertIn('--strict-mcp-config', args)
            self.assertNotIn('User speech on stdin.', args)
            self.assertEqual((self.root / 'input').read_text(), 'User speech on stdin.')
            (self.root / 'gate').touch()
            self.assertEqual([part async for part in turn], ['Ready.'])
            self.assertIn('onju-8DAB80', self.store.data)
            self.assert_reaped()
        finally:
            await turn.aclose()

    async def test_sse_response_yields_before_child_completes_and_close_reaps(self):
        # Exercise the route's nested generators, without a buffering TestClient.
        app = voice_agent.create_app(self.settings)
        body = json.dumps({'user': 'onju-8DAB80', 'stream': True,
                           'messages': [{'role': 'user', 'content': 'hi'}]}).encode()
        async def receive():
            return {'type': 'http.request', 'body': body, 'more_body': False}
        request = Request({'type': 'http', 'method': 'POST', 'path': '/v1/chat/completions',
                           'headers': []}, receive)
        endpoint = next(r.endpoint for r in app.routes if r.path == '/v1/chat/completions')
        response = await asyncio.wait_for(endpoint(request), 1)
        stream = response.body_iterator
        first = await asyncio.wait_for(anext(stream), 3)
        self.assertEqual(json.loads(first.removeprefix('data: '))['choices'][0]
                         ['delta']['content'], 'Hello. ')
        self.assertFalse((self.root / 'finished').exists())
        await stream.aclose()
        self.assert_reaped()
        self.assertEqual(json.loads(self.settings.state_file.read_text()), {})

    async def test_http_client_disconnect_reaps_running_child(self):
        app = voice_agent.create_app(self.settings)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(3):
                while not server.started:
                    await asyncio.sleep(.01)
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                                         json={"user": "onju-8DAB80", "stream": True,
                                               "messages": [{"role": "user", "content": "hi"}]}) as r:
                    self.assertEqual(r.status_code, 200)
                    first = await asyncio.wait_for(anext(r.aiter_lines()), 3)
                    self.assertIn("Hello. ", first)
                    self.assertFalse((self.root / "finished").exists())
                async with asyncio.timeout(3):
                    while True:
                        try:
                            os.kill(int((self.root / "pid").read_text()), 0)
                        except ProcessLookupError:
                            break
                        await asyncio.sleep(.01)
                self.assertEqual(json.loads(self.settings.state_file.read_text()), {})
                # A following request proves the per-device lock was released.
                os.environ["STREAM_TEST_MODE"] = "ok"
                r = await client.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                                      json={"user": "onju-8DAB80", "stream": True,
                                            "messages": [{"role": "user", "content": "next"}]})
                self.assertIn("Ready.", r.text)
                self.assertTrue(r.text.rstrip().endswith("data: [DONE]"))
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 5)
            listener.close()

    async def test_timeout_after_partial_output_reaps_and_drops_persisted_session(self):
        self.store.touch('onju-8DAB80', 'old-session', voice_agent.time.time())
        self.settings.timeout = .3
        turn = self.turn()
        self.assertEqual(await asyncio.wait_for(anext(turn), 3), 'Hello. ')
        rest = await asyncio.wait_for(anext(turn), 3)
        self.assertEqual(rest, '\n' + voice_agent.FALLBACK_REPLY)
        self.assert_reaped()
        self.assertEqual(json.loads(self.settings.state_file.read_text()), {})
        with self.assertRaises(StopAsyncIteration):
            await anext(turn)

    async def test_cancel_waiting_for_next_delta_reaps_and_releases_lock(self):
        turn = self.turn()
        self.assertEqual(await asyncio.wait_for(anext(turn), 3), 'Hello. ')
        waiter = asyncio.create_task(anext(turn))
        await asyncio.sleep(.03)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assert_reaped()
        self.assertEqual(self.store.data, {})
        # The same device can start a subsequent turn after cancellation.
        os.environ['STREAM_TEST_MODE'] = 'ok'
        self.assertEqual(''.join([part async for part in self.turn()]), 'Hello. Ready.')
        args = json.loads((self.root / 'args').read_text())
        self.assertIn('--session-id', args)

    async def test_partial_failure_no_result_and_malformed_output_are_not_success(self):
        for mode in ('failure', 'noresult', 'malformed'):
            with self.subTest(mode=mode):
                os.environ['STREAM_TEST_MODE'] = mode
                output = [part async for part in self.turn()]
                self.assertEqual(output, ['Hello. ', '\n' + voice_agent.FALLBACK_REPLY])
                self.assertEqual(self.store.data, {})
                self.assert_reaped()

    async def test_large_stderr_does_not_block_or_leak_and_success_resumes(self):
        os.environ['STREAM_TEST_MODE'] = 'stderr'
        output = await asyncio.wait_for(self.collect(), 4)
        self.assertEqual(output, 'Hello. Ready.')
        sid = self.store.data['onju-8DAB80']['id']
        output = await asyncio.wait_for(self.collect(), 4)
        args = json.loads((self.root / 'args').read_text())
        self.assertEqual(args[args.index('--resume') + 1], sid)
        self.assert_reaped()

    async def collect(self):
        return ''.join([part async for part in self.turn()])


if __name__ == '__main__':
    unittest.main()
