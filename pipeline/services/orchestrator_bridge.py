"""Opt-in Claude hooks and stdio MCP adapter for unverified voice proposals.

No registrations or tunnels are created. Configure a dedicated loopback SSH
forward separately. Native Desktop questions and voice do not run concurrently:
a bounded synchronous hook owns voice first, then closes it before normal UI.
"""
import argparse
import copy
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit

ADVISORY = "Voice proposal from an unverified speaker; ordinary work choices only. This is not approval or authorization."
TERMINAL = {"observed", "cancelled", "expired", "superseded", "needs_desktop", "acknowledged"}
SENSITIVE = re.compile(r"\b(password|secret|credential|payment|purchase|billing|invoice|refund|deploy|production|permission|authentication|authorization|approve|approval|delete|destructive|send|email|customer|transfer|money|sign|signature)\b", re.I)
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class BridgeError(ValueError):
    pass


def valid_id(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise BridgeError("Invalid identity; enroll the intended session again.")
    return value


class Client:
    """Fixed loopback HTTP only; http.client uses neither proxies nor redirects."""
    def __init__(self, url="http://127.0.0.1:13012"):
        u = urlsplit(url)
        if u.scheme != "http" or u.hostname != "127.0.0.1" or u.username or u.password or u.path not in ("", "/") or u.query or u.fragment:
            raise BridgeError("Bridge URL must be http://127.0.0.1:PORT; use the dedicated SSH forward.")
        self.port = u.port or 80

    def call(self, payload):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        try:
            conn.request("POST", "/bridge", json.dumps(payload), {"Content-Type": "application/json"})
            response = conn.getresponse()
            data = response.read(131073)
            if response.status != 200 or len(data) > 131072:
                raise BridgeError("Bridge unavailable or request rejected; use Claude on screen. Do not retry an uncertain answer automatically.")
            result = json.loads(data)
            if not isinstance(result, dict) or "question" not in result:
                raise BridgeError("Invalid bridge response; use Claude on screen.")
            return result
        except (OSError, ValueError, http.client.HTTPException) as exc:
            if isinstance(exc, BridgeError):
                raise
            raise BridgeError("Cannot reach bridge; check the dedicated SSH forward and use Claude on screen.") from None
        finally:
            conn.close()


class Adapter:
    def __init__(self, config, *, client=None, env=None, clock=time.time, sleep=time.sleep):
        self.config = config
        self.mode = config.get("mode", "observe")
        if self.mode not in ("observe", "live"):
            raise BridgeError("Bridge mode must be observe or live.")
        self.host = valid_id(config.get("host_session_id"))
        self.env = os.environ if env is None else env
        if self.env.get("CLAUDE_CODE_HOST_SESSION_ID") != self.host:
            raise BridgeError("Host does not match the pinned Orchestrator; no routing performed.")
        if config.get("device") != "onju-8DAB80":
            raise BridgeError("Bridge device must be pinned to onju-8DAB80.")
        self.client = client or Client(config.get("url", "http://127.0.0.1:13012"))
        self.clock, self.sleep = clock, sleep
        self.leases = {}
        self.timer = time.monotonic if clock is time.time else clock
        self.state = Path(config.get("state_dir", "~/.local/state/aiden-orchestrator")).expanduser()
        self.state.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.state.is_symlink() or self.state.stat().st_uid != os.getuid() or self.state.stat().st_mode & 0o077:
            raise BridgeError("Bridge state directory must be private and owned by this user.")

    def write(self, name, value):
        dest = self.state / name
        if not dest.exists() and sum(1 for _ in self.state.iterdir()) >= 1024:
            raise BridgeError("Private bridge storage is full; reconcile existing receipts before routing more questions.")
        tmp = self.state / (name + "." + str(os.getpid()) + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(value, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)

    def binding(self):
        session = self.config.get("session_id")
        if session:
            return valid_id(session)
        try:
            data = json.loads((self.state / "binding.json").read_text())
        except (OSError, ValueError):
            raise BridgeError("MCP session is not bound. Run the prepared SessionStart enrollment hook in the intended conversation, then restart this MCP server.") from None
        if data.get("host_session_id") != self.host or data.get("verified_hook") is not True:
            raise BridgeError("MCP binding does not match the pinned host; enroll again.")
        return valid_id(data.get("session_id"))

    def identity(self, session, question_id, version=1):
        return {"host_session_id": self.host, "session_id": valid_id(session), "question_id": valid_id(question_id), "version": version}

    def call(self, op, identity, **fields):
        key = (identity.get("session_id"), identity.get("question_id"), identity.get("version"))
        try:
            return self.client.call({"op": op, **identity, **fields})
        finally:
            if op in ("submit", "renew"):
                # Even a failed response may follow an accepted server mutation.
                # Latest possible server lease start is bounded by call completion.
                self.leases[key] = self.timer() + 12  # 10s lease + 2s margin

    def observe(self, identity, kind):
        # Metadata only: neither question/answer text nor any remote mutation.
        self.write("observe-" + identity["question_id"] + ".json", {**identity, "kind": kind, "status": "observed", "would_announce": True, "at": self.clock()})
        return {"status": "observed", "question_id": identity["question_id"], "version": identity["version"]}

    def close(self, identities, op="cancel"):
        uncertain_until = self.timer()
        for identity in identities:
            key = (identity.get("session_id"), identity.get("question_id"), identity.get("version"))
            try:
                result = self.call(op, identity)
                if result["question"].get("status") not in TERMINAL:
                    uncertain_until = max(uncertain_until, self.leases.get(key, self.timer() + 12))
            except BridgeError:
                uncertain_until = max(uncertain_until, self.leases.get(key, self.timer() + 12))
        # Never release native UI while an uncertain voice lease could be live.
        while self.timer() < uncertain_until:
            self.sleep(min(.25, uncertain_until - self.timer()))

    def wait(self, identity, deadline, renew_identities=None):
        renew_at = self.clock() + 3
        while self.clock() < deadline:
            if self.clock() >= renew_at:
                for pending in renew_identities or [identity]:
                    self.call("renew", pending)
                renew_at = self.clock() + 3
            q = self.call("status", identity)["question"]
            if not isinstance(q, dict):
                raise BridgeError("Question is unavailable; use Claude on screen.")
            if any(q.get(key) != identity[key] for key in ("host_session_id", "session_id", "question_id", "version")):
                return None
            if q.get("status") == "captured":
                return q
            if q.get("status") in TERMINAL:
                return None
            self.sleep(min(.25, max(0, deadline - self.clock())))
        return None

    def submit(self, identity, kind, text, options, requires_desktop=False, summary="", deadline=None):
        desktop = requires_desktop or bool(SENSITIVE.search(text + " " + summary + " " + json.dumps(options)))
        return self.call("submit", identity, kind=kind, tool_use_id=identity.get("tool_use_id", identity["question_id"]), text=text, options=options, **({"summary": summary} if summary else {}), expires_at=deadline or self.clock() + 55, requires_desktop=desktop, device=self.config["device"])

    def receipt_name(self, session, tool):
        return "receipt-" + hashlib.sha256((self.host + ":" + session + ":" + tool).encode()).hexdigest() + ".json"

    def hook(self, event):
        session = valid_id(event.get("session_id"))
        if event.get("hook_event_name") == "SessionStart":
            def directory_metadata(value):
                # Retain only an explicit absolute directory supplied by the
                # hook/runtime. A moving cwd is never a project-root fallback.
                if value is None:
                    return None
                if not isinstance(value, str) or len(value) > 4096 or "\x00" in value or not Path(value).is_absolute() or not Path(value).is_dir():
                    raise BridgeError("Enrollment directory metadata must identify an existing absolute directory.")
                return value
            self.write("binding.json", {"host_session_id": self.host, "session_id": session, "verified_hook": True,
                                       "cwd": directory_metadata(event.get("cwd")),
                                       "project_dir": directory_metadata(self.env.get("CLAUDE_PROJECT_DIR"))})
            return {}
        if event.get("tool_name") != "AskUserQuestion":
            return {}
        tool = valid_id(event.get("tool_use_id"))
        receipt_name = self.receipt_name(session, tool)
        if event.get("hook_event_name") == "PostToolUse":
            try:
                receipt = json.loads((self.state / receipt_name).read_text())
            except (OSError, ValueError):
                return {}
            # Both the original questions and resulting answers must match.
            response = event.get("tool_response")
            if receipt.get("status") != "emitted" or event.get("tool_input", {}).get("questions") != receipt["questions"] or not isinstance(response, dict) or response.get("answers") != receipt["answers"]:
                return {}
            for item in receipt["items"]:
                self.call("ack", item["identity"], answer_id=item["answer_id"])
            return {}
        if event.get("hook_event_name") != "PreToolUse":
            return {}
        original = event.get("tool_input")
        if not isinstance(original, dict):
            return {}
        questions = original.get("questions")
        if not isinstance(questions, list) or not 1 <= len(questions) <= 4:
            return {}
        # Validate the entire batch before speaking any of it. The answers map
        # cannot represent repeated question text with different answers.
        if len({q.get("question") for q in questions if isinstance(q, dict) and isinstance(q.get("question"), str)}) != len(questions):
            return {}
        for question in questions:
            if not isinstance(question, dict) or question.get("multiSelect") or question.get("requires_desktop") or "preview" in question:
                return {}
            if SENSITIVE.search(json.dumps(question)):
                return {}
            choices = question.get("options")
            if not isinstance(question.get("question"), str) or not question["question"].strip() or len(question["question"]) > 2000:
                return {}
            if not isinstance(choices, list) or not 2 <= len(choices) <= 8 or any(not isinstance(o, dict) or set(o) - {"label", "description"} or not isinstance(o.get("label"), str) or not o["label"].strip() or len(o["label"]) > 120 or not isinstance(o.get("description", ""), str) or len(o.get("description", "")) > 500 for o in choices):
                return {}
        identities, captures = [], []
        answers = {}
        emitted = False
        deadline = self.clock() + 55
        if self.mode == "live":
            # Reserve bounded private storage before any remote submit. An
            # interrupted reservation remains uncertain until reconciled.
            self.write(receipt_name, {"status": "pending", "session_id": session, "tool_use_id": tool})
        try:
            for index, q in enumerate(questions):
                if not isinstance(q, dict) or not isinstance(q.get("question"), str) or len(q["question"]) > 2000:
                    return {}
                options = q.get("options")
                if q.get("multiSelect") or q.get("preview") or not isinstance(options, list) or not 2 <= len(options) <= 8 or any(not isinstance(o, dict) or not isinstance(o.get("label"), str) or not isinstance(o.get("description", ""), str) or len(o["label"]) > 120 or len(o.get("description", "")) > 500 or "preview" in o for o in options):
                    return {}
                qid = hashlib.sha256(f"{self.host}:{session}:{tool}:{index}".encode()).hexdigest()
                identity = {**self.identity(session, qid), "tool_use_id": tool}
                identities.append(identity)
                if self.mode == "observe":
                    self.observe(identity, "structured")
                    continue
                self.submit(identity, "structured", q["question"], options, bool(q.get("requires_desktop")), deadline=deadline)
                captured = self.wait(identity, deadline, identities)
                if captured is None:
                    return {}
                answer = captured.get("answer", {})
                choice = answer.get("choice")
                if type(choice) is int and 1 <= choice <= len(options):
                    value = options[choice - 1]["label"]
                elif choice is None and isinstance(answer.get("text"), str) and answer["text"].strip():
                    value = answer["text"]
                else:
                    return {}
                answers[q["question"]] = value
                captures.append({"identity": identity, "answer_id": valid_id(captured.get("answer_id"))})
            if self.mode == "observe":
                return {}
            if self.clock() >= deadline:
                return {}
            for item in captures:
                result = self.call("emitted", item["identity"], answer_id=item["answer_id"])
                if result["question"].get("status") != "emitted":
                    return {}
            self.write(receipt_name, {"status": "emitted", "questions": questions, "answers": answers, "items": captures})
            emitted = True
            updated = copy.deepcopy(original)
            updated["answers"] = answers
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow", "updatedInput": updated, "additionalContext": ADVISORY}}
        finally:
            if self.mode == "live" and not emitted:
                self.close(identities)

    def question(self, args):
        session = self.binding()
        qid = valid_id(args.get("question_id"))
        version = args.get("version", 1)
        if type(version) is not int or version < 1:
            raise BridgeError("Version must be a positive integer.")
        text = args.get("text")
        summary = args.get("summary", "")
        options = args.get("options", [])
        if not isinstance(text, str) or not text.strip() or len(text) > 2000 or not isinstance(summary, str) or len(summary) > 500 or not isinstance(options, list) or len(options) > 8:
            raise BridgeError("Use one question up to 2000 characters, summary up to 500, and at most eight options.")
        for option in options:
            if not isinstance(option, dict) or set(option) - {"label", "description"} or not isinstance(option.get("label"), str) or not option["label"].strip() or len(option["label"]) > 120 or not isinstance(option.get("description", ""), str) or len(option.get("description", "")) > 500:
                raise BridgeError("Each option needs a label up to 120 characters and description up to 500; visual choices require Claude on screen.")
        identity = self.identity(session, qid, version)
        if self.mode == "observe":
            return {**self.observe(identity, "explicit"), "notice": "Observe only: no audio or answer routing."}
        completed = False
        try:
            deadline = self.clock() + 55
            result = self.submit(identity, "explicit", text, options, bool(args.get("requires_desktop")), summary, deadline)
            if result["question"].get("requires_desktop"):
                self.call("screen", identity)
                return {**identity, "status": "needs_desktop", "notice": "This question requires Claude on screen; voice cannot authorize it."}
            captured = self.wait(identity, deadline)
            if captured is None:
                return {**identity, "status": "needs_desktop", "notice": "Voice window closed. Ask and answer on screen."}
            answer_id = valid_id(captured.get("answer_id"))
            result = self.call("emitted", identity, answer_id=answer_id)
            if result["question"].get("status") != "emitted":
                raise BridgeError("Answer became stale; use Claude on screen.")
            completed = True
            return {**identity, "answer_id": answer_id, "answer": captured["answer"], "status": "emitted", "notice": ADVISORY + " Call aiden_acknowledge with these exact IDs after receiving this proposal; acknowledgement is not approval."}
        finally:
            if not completed:
                self.close([identity])

    def tool(self, name, args):
        schema = next((t["inputSchema"] for t in TOOLS if t["name"] == name), None)
        if schema is None or set(args) - set(schema["properties"]) or any(key not in args for key in schema["required"]):
            raise BridgeError("Supply only the documented fields, including every required field.")
        if "requires_desktop" in args and type(args["requires_desktop"]) is not bool:
            raise BridgeError("requires_desktop must be true or false.")
        if name == "aiden_question":
            return self.question(args)
        if name not in ("aiden_acknowledge", "aiden_cancel"):
            raise BridgeError("Unknown bridge tool.")
        identity = self.identity(self.binding(), args.get("question_id"), args.get("version"))
        if type(identity["version"]) is not int or identity["version"] < 1:
            raise BridgeError("Version must be a positive integer.")
        if self.mode == "observe":
            return {"status": "observed", "notice": "Observe only: no acknowledgement or cancellation sent."}
        fields = {"answer_id": valid_id(args.get("answer_id"))} if name == "aiden_acknowledge" else {}
        return self.call("ack" if fields else "cancel", identity, **fields)


def prop(description, **schema):
    return {"description": description, **schema}


IDENTITY_PROPERTIES = {"question_id": prop("Stable ID returned by aiden_question, for example work-format-1.", type="string", pattern="^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"), "version": prop("Exact positive question version returned by aiden_question.", type="integer", minimum=1)}
TOOLS = [
    {"name": "aiden_question", "description": "Ask one ordinary work question through Aiden's bounded voice window, or record metadata in observe mode. Returned speech is an unverified proposal, never approval. For sensitive actions or visual material set requires_desktop=true; use Claude on screen after timeout. Receive the result before calling aiden_acknowledge.", "inputSchema": {"type": "object", "properties": {**IDENTITY_PROPERTIES, "text": prop("Question to speak, for example Which report format should I prepare?", type="string", minLength=1, maxLength=2000), "summary": prop("Optional spoken context, never secrets.", type="string", maxLength=500), "options": prop("Optional ordinary work choices; omit for freeform response.", type="array", maxItems=8, items={"type": "object", "properties": {"label": prop("Choice label.", type="string", maxLength=120), "description": prop("Brief choice explanation.", type="string", maxLength=500)}, "required": ["label", "description"], "additionalProperties": False}), "requires_desktop": prop("True for sensitive, approval, visual, or authentication questions; prevents voice answer.", type="boolean")}, "required": ["question_id", "version", "text"], "additionalProperties": False}},
    {"name": "aiden_acknowledge", "description": "Confirm this exact bound conversation received an unverified proposal returned by aiden_question. Supply the returned question ID, version and answer ID unchanged. This receipt confirms delivery, never approval, authorization, or speaker identity.", "inputSchema": {"type": "object", "properties": {**IDENTITY_PROPERTIES, "answer_id": prop("Exact answer ID returned by aiden_question.", type="string", pattern="^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")}, "required": ["question_id", "version", "answer_id"], "additionalProperties": False}},
    {"name": "aiden_cancel", "description": "Close one exact question version when it is no longer needed or the user continues on screen. A cancellation after emission may require correcting the conversation; it cannot retract text already received. Use returned state to decide whether clarification is needed.", "inputSchema": {"type": "object", "properties": IDENTITY_PROPERTIES, "required": ["question_id", "version"], "additionalProperties": False}},
]


def handle(adapter, msg):
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid JSON-RPC request"}}
    if "id" not in msg:
        return None
    mid, method = msg["id"], msg.get("method")
    params = msg.get("params", {})
    if not isinstance(params, dict):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "params must be an object"}}
    if method == "initialize":
        requested = params.get("protocolVersion")
        version = requested if requested in ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25") else "2025-11-25"
        result = {"protocolVersion": version, "capabilities": {"tools": {}}, "serverInfo": {"name": "aiden-orchestrator", "version": "1.0"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "ping":
        result = {}
    elif method == "tools/call":
        name, args = params.get("name"), params.get("arguments", {})
        if name not in {t["name"] for t in TOOLS}:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "Unknown tool"}}
        try:
            if not isinstance(args, dict):
                raise BridgeError("Tool arguments must be an object.")
            value = adapter.tool(name, args)
            result = {"content": [{"type": "text", "text": json.dumps(value)}], "isError": False}
        except BridgeError as exc:
            result = {"content": [{"type": "text", "text": str(exc)}], "isError": True}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "Unknown method"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("command", choices=("hook", "enroll", "mcp"))
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).expanduser().read_text())
        if not isinstance(config, dict):
            raise BridgeError("Bridge config must be an object.")
        adapter = Adapter(config)
        if args.command in ("hook", "enroll"):
            event = json.load(sys.stdin)
            if args.command == "enroll" and event.get("hook_event_name") != "SessionStart":
                raise BridgeError("Enrollment requires verified SessionStart hook input.")
            print(json.dumps(adapter.hook(event)))
        else:
            adapter.binding()  # Bind once at startup, never trust caller session IDs.
            adapter.config = {**config, "session_id": adapter.binding()}
            for line in sys.stdin:
                try:
                    msg = json.loads(line)
                    reply = handle(adapter, msg)
                except ValueError:
                    reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
                if reply is not None:
                    print(json.dumps(reply), flush=True)
    except (BridgeError, OSError, ValueError) as exc:
        # Never echo configuration, response bodies or incoming question text.
        print("Aiden bridge unavailable; use Claude on screen.", file=sys.stderr)
        if args.command in ("hook", "enroll"):
            print("{}")
        else:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
