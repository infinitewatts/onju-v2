"""Bounded, opt-in connection test; never a channel for arbitrary commands.

IDs and results live only for this process. At most sixteen IDs are accepted;
new IDs are rejected at capacity rather than evicting an idempotency record.
A restart invalidates all prior IDs. Status describes local processing, never
proof that a person heard a prompt or that a particular person answered it.
"""
from dataclasses import dataclass
import re
import time

QUESTION = (
    "Orchestrator connection test. Choose option one, blue, or option two, green. "
    "Say Hey Aiden, followed by your choice."
)
CLARIFICATION = "Please say Hey Aiden, option one for blue, or option two for green."
INACTIVE = "That connection test is no longer active."
COMPLETE = "That connection test is already complete."
CONFIRMATIONS = {1: "Recorded option one, blue, for this connection test.",
                 2: "Recorded option two, green, for this connection test."}
LABELS = {1: "blue", 2: "green"}
PENDING = frozenset(("queued", "speaking", "awaiting_answer"))
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
DEVICE_RE = re.compile(r"onju-[0-9A-F]{6}\Z")
TTL_S = 300
MAX_RESULTS = 16


class PilotError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class Question:
    id: str
    host_session_id: str
    device: str
    deadline: float
    status: str = "queued"
    choice: int | None = None

    def snapshot(self) -> dict:
        result = {"id": self.id, "host_session_id": self.host_session_id,
                  "device": self.device, "status": self.status}
        if self.choice is not None:
            result.update(choice=self.choice, label=LABELS[self.choice])
        return result


class VoicePilot:
    def __init__(self, config: dict, clock=time.monotonic):
        self.enabled = config.get("enabled", False) is True
        self.host_session_id = config.get("host_session_id")
        if self.enabled and (not isinstance(self.host_session_id, str)
                             or not ID_RE.fullmatch(self.host_session_id)):
            raise ValueError("Enabled voice pilot requires a conservative host_session_id")
        self.clock = clock
        self.records: dict[str, Question] = {}

    def _require_enabled(self):
        if not self.enabled:
            raise PilotError(404, "Voice pilot is disabled")

    def _expire(self):
        now = self.clock()
        for item in self.records.values():
            if item.status in PENDING and now >= item.deadline:
                item.status = "expired"

    def validate(self, payload: dict):
        self._require_enabled()
        if (not isinstance(payload, dict)
                or set(payload) != {"id", "host_session_id", "device"}
                or any(not isinstance(value, str) for value in payload.values())
                or not ID_RE.fullmatch(payload["id"])
                or not ID_RE.fullmatch(payload["host_session_id"])
                or not DEVICE_RE.fullmatch(payload["device"])):
            raise PilotError(400, "Expected only valid id, host_session_id and device")
        if payload["host_session_id"] != self.host_session_id:
            raise PilotError(409, "Host session does not match this pilot")

    def submit(self, payload: dict, device: str, *, busy: bool) -> tuple[dict, bool]:
        self.validate(payload)
        if payload["device"] != device:
            raise PilotError(409, "Pilot device does not match")
        self._expire()
        previous = self.records.get(payload["id"])
        if previous:
            if previous.device != device or previous.host_session_id != payload["host_session_id"]:
                raise PilotError(409, "Question ID already belongs to another target")
            return previous.snapshot(), False
        if any(item.status in PENDING for item in self.records.values()):
            raise PilotError(409, "A connection test is already pending")
        if busy:
            raise PilotError(409, "Pilot device is busy")
        if len(self.records) >= MAX_RESULTS:
            raise PilotError(409, "Pilot ID capacity reached; restart invalidates prior IDs")
        item = Question(payload["id"], self.host_session_id, device, self.clock() + TTL_S)
        self.records[item.id] = item
        return item.snapshot(), True

    def get(self, question_id: str) -> dict:
        self._require_enabled()
        if not isinstance(question_id, str) or not ID_RE.fullmatch(question_id):
            raise PilotError(400, "Invalid question ID")
        self._expire()
        item = self.records.get(question_id)
        if item is None:
            raise PilotError(404, "Unknown question ID")
        return item.snapshot()

    def can_send(self, question_id: str) -> bool:
        self._expire()
        item = self.records.get(question_id)
        return item is not None and item.status in ("queued", "speaking")

    def speaking(self, question_id: str) -> bool:
        if not self.can_send(question_id):
            return False
        self.records[question_id].status = "speaking"
        return True

    def awaiting_answer(self, question_id: str):
        if self.can_send(question_id):
            self.records[question_id].status = "awaiting_answer"

    def fail(self, question_id: str):
        self._expire()
        item = self.records.get(question_id)
        if item is not None and item.status in PENDING:
            item.status = "failed"

    @staticmethod
    def parse_choice(text: str) -> int | None:
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        normalized = re.sub(r"^(?:hey )?aiden\b\s*", "", normalized)
        return {"one": 1, "1": 1, "option one": 1, "option 1": 1, "blue": 1,
                "two": 2, "2": 2, "option two": 2, "option 2": 2,
                "green": 2}.get(normalized)

    def intercept(self, device: str, text: str) -> str | None:
        """Return fixed speech to bypass the LLM; never retain raw recognition."""
        if not self.enabled:
            return None
        self._expire()
        choice = self.parse_choice(text)
        matching = [item for item in self.records.values() if item.device == device]
        if not matching:
            return None
        current = matching[-1]
        if current.status == "awaiting_answer":
            if choice is None:
                return CLARIFICATION
            current.choice = choice
            current.status = "answered"
            return CONFIRMATIONS[choice]
        if current.status == "answered" and choice is not None:
            return COMPLETE
        if current.status in ("expired", "failed") and choice is not None:
            return INACTIVE
        return None
