# Orchestrator voice connection pilot

This opt-in test connects one fixed harmless question to a verified Claude
Desktop Code host session. It is not standing automation or an approval channel.
The normal voice agent and its restricted tools remain separate.

## Before enabling

Obtain the host ID from `CLAUDE_CODE_HOST_SESSION_ID` inside the intended Desktop
conversation and compare it to the department mapping. Do not infer the target
from a title alone or resume a separate CLI conversation as a substitute.
Obtain owner approval for any wake-bridge allowlist change. Keep the existing
configuration and prior reviewed deployment SHA for rollback.

Enable only in the Mini's local `pipeline/config.yaml`, after review:

```yaml
voice_pilot:
  enabled: true
  host_session_id: local_VERIFIED_HOST_ID
```

The checked-in deployment template does not enable this pilot. The control
server stays on loopback; access it from the MacBook through the existing SSH
connection. No new listening ports or credentials are required.

## Test protocol

1. From the verified Orchestrator conversation, POST `/pilot/question` to the
   Mini's control port with JSON fields `id`, `host_session_id`, and `device`.
   Use a unique conservative test ID and the configured speaker hostname.
   The content is fixed: option one is blue, option two is green.
2. Read `/pilot/question?id=TEST_ID` to establish that the question is awaiting
   an answer. HTTP acceptance is not proof that a human heard playback.
3. The owner says "Hey Aiden, option one" or "Hey Aiden, option two". The local
   pipeline interprets only the fixed choices. The question expires after five
   minutes. A pipeline restart invalidates in-memory test state.
4. Fetch the correlated result. Send only its test ID and fixed choice/label
   through the reviewed wake sender to the same verified Desktop host.
   This is unverified-speaker test data and grants no authority.
5. Require an explicit acknowledgement from that exact conversation, including
   the test ID and selected choice. A wake file disappearing does not establish
   delivery. Prove idle-session delivery separately from delivery during a turn.

Only one question can be active. Busy devices and overlapping questions must
be rejected rather than interrupting existing speech. Repeating an existing ID
returns its existing state without replaying the question; use a new ID for a
deliberate retry. At most sixteen test IDs are retained per process; at capacity,
new tests are rejected rather than forgetting an ID and accidentally replaying
it. A restart clears those IDs. There is no background forwarding process in
this pilot.

## End and recovery

Disable `voice_pilot` and restart only the pipeline after the test. Restore the
prior configuration/SHA and restart the pipeline if startup health, device
registration, or control health fails after deployment. Preserve unrelated
configuration changes made after the backup; never blindly restore over them.

Wake-bridge cleanup must account for the installed hook's lifecycle. Removing
a host from a configuration file is not sufficient if an already-running hook
only reads that file at startup. Verify actual watcher shutdown or use a reviewed
revocable mechanism before calling the permission reverted. Do not alter other
managers' schedules or routing as part of this pilot.
