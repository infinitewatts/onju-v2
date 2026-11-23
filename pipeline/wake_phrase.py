"""The one list of spellings that count as the wake phrase.

The wake gate, the command prefix strip and the pilot choice parser all read
these, so a spelling the gate admits is also removed before routing.
"""
import re

# Spellings the recognizer uses for the same two sounds; no other names.
GREETING = r"(?:hey|hay)"
NAME = r"(?:aiden|aidan|ayden|aydan|aidyn|aydin|aden)"
_COMMAND_PREFIX = re.compile(rf"^\s*(?:{GREETING}[\s,.!?:;-]+)?{NAME}\b[\s,.!?;:-]*", re.IGNORECASE)


def strip_wake_prefix(text):
    """Remove a leading wake phrase (greeting optional) from command text."""
    return _COMMAND_PREFIX.sub("", text).strip()
