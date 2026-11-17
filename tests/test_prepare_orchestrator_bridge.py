import hashlib
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from scripts.prepare_orchestrator_bridge import prepare, SOURCE


class PreparationTests(unittest.TestCase):
    def test_private_exact_copy_observe_fragments_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "bundle with spaces"
            receipt = prepare(output, "local_test-host", "a" * 40)
            self.assertEqual((output / "adapter.py").read_bytes(), SOURCE.read_bytes())
            config = json.loads((output / "adapter.json").read_text())
            self.assertEqual(config["mode"], "observe")
            self.assertNotIn("session_id", config)
            hooks = json.loads((output / "claude-hooks.fragment.json").read_text())["hooks"]
            command = hooks["SessionStart"][0]["hooks"][0]["command"]
            self.assertEqual(shlex.split(command)[1], str(output / "adapter.py"))
            self.assertEqual(set(hooks), {"SessionStart", "PreToolUse", "PostToolUse"})
            mcp = json.loads((output / "claude-mcp.fragment.json").read_text())["mcpServers"]["aiden-orchestrator"]
            self.assertEqual(mcp["args"][-1], "mcp")
            forward = json.loads((output / "ssh-forward.argv.json").read_text())
            self.assertIn("-oExitOnForwardFailure=yes", forward)
            self.assertIn("-L127.0.0.1:13012:127.0.0.1:3012", forward)
            for name, digest in receipt["files"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            for child in output.iterdir():
                self.assertEqual(child.stat().st_mode & 0o777, 0o700 if child.is_dir() else 0o600)
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
    def test_refuses_existing_output_invalid_identity_or_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "bundle"
            prepare(output, "local_host")
            before = (output / "adapter.py").read_bytes()
            with self.assertRaises(ValueError):
                prepare(output, "local_host")
            self.assertEqual((output / "adapter.py").read_bytes(), before)
            with self.assertRaises(ValueError):
                prepare(Path(temp) / "bad", "../../host")
            with self.assertRaises(ValueError):
                prepare(Path(temp) / "bad", "local_host", "notsha")

    def test_staging_can_render_future_runtime_without_installing(self):
        with tempfile.TemporaryDirectory() as temp:
            output, runtime = Path(temp) / "staging", Path(temp) / "runtime"
            receipt = prepare(output, "local_host", runtime_dir=runtime)
            self.assertFalse(runtime.exists())
            config = json.loads((output / "adapter.json").read_text())
            self.assertEqual(config["state_dir"], str(runtime / "state"))
            mcp = json.loads((output / "claude-mcp.fragment.json").read_text())["mcpServers"]["aiden-orchestrator"]
            self.assertEqual(mcp["args"][0], str(runtime / "adapter.py"))
            self.assertEqual(receipt["runtime_dir"], str(runtime))
