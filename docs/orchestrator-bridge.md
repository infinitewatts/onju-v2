# Aiden–Orchestrator bridge

This bridge is opt-in. The default pipeline configuration leaves it off. Its
prepared Claude adapter starts in **observe** mode; preparing files does not
register hooks, install an MCP server, open a tunnel, or enable announcements.

## Supported workflow

Claude Desktop Code supports a synchronous `PreToolUse` hook that supplies
`AskUserQuestion` answers through `updatedInput` before the native question UI
opens. This bridge uses that supported **voice-first** window. It does not inject
answers into an already-open desktop question. Saying “on screen,” an expired
window, or a failed bridge hands control back to the original Claude interface.
Original questions are preserved. A batch shares one bounded answer window.

`PostToolUse` acknowledges only a matching correlated result. Cancellation does
not reliably produce a post-tool hook, so leases, expiry, and question versions
reject stale voice answers. A missing acknowledgement stays unconfirmed; it
never becomes “Orchestrator received it” merely because a queue was consumed.

Ordinary questions use the explicit `aiden_question` MCP tool. They are not
inferred from punctuation or scraped conversation history. After receiving an
answer proposal, Orchestrator calls `aiden_acknowledge` with the returned identity
and answer ID. That receipt permits the delivery confirmation. `aiden_cancel`
withdraws an undelivered question; after emission, cancellation is a correction
request, not a claim that an answer was unsent.

The installed Desktop engine and the standalone CLI may differ. Interfaces were
checked against official documentation; actual Desktop enrollment and live
round-trip acceptance remain required before claiming operation in the intended
conversation.

References:
- https://code.claude.com/docs/en/hooks#tools-that-require-user-interaction
- https://code.claude.com/docs/en/hooks#posttoolusefailure
- https://code.claude.com/docs/en/desktop#claude-code-version-in-the-code-tab

## Routing and authority

The adapter checks `CLAUDE_CODE_HOST_SESSION_ID` against its configured host. A
SessionStart enrollment records the concrete Claude session ID in a private
binding file. The MCP process freezes that binding at startup; restart it after
enrollment changes. Titles are never routing identifiers. The previously verified
Orchestrator host is `local_043ec9c0-8316-4e28-8304-5b405c514315`; a fresh enrollment
must establish the current concrete session before live use.

Use project-local MCP registration from the verified primary project directory,
not user-wide or Desktop Chat registration. Local scope lives under that project's
entry in `~/.claude.json`; use the supported `claude mcp add-json --scope local`
command, preserving existing servers. SessionStart enrollment retains available
absolute `cwd` and `CLAUDE_PROJECT_DIR` metadata; verify the primary project rather
than guessing from a chat title or treating a transient working directory as its
root. Local scope isolates projects, not conversations: another chat in that
project may discover the entry, but its different host fails the adapter pin.
No project path is preselected in the prepared bundle.

Desktop local Code shares this configuration as documented at
https://code.claude.com/docs/en/desktop#shared-configuration and
https://code.claude.com/docs/en/mcp#local-scope .

All speech is an **unverified proposal**, not speaker authentication or permission
to perform sensitive actions. Sensitive or visual questions belong on screen.
The bridge does not register permission-request approval handlers. Do not use
voice replies to bypass existing approval requirements for money, credentials,
permissions, customer communications, or production changes.

Only one question awaits a voice answer at a time. Captured answers can remain
pending acknowledgement while the next question is presented. Every mutation is
correlated with host, session, question ID, and version. Answers have their own
stable IDs. Durable receipt claims prevent duplicate delivery announcements after
restart; playback failure can omit a confirmation rather than announce it twice.

## Spoken interaction

After an announcement, say “Hey Aiden,” then:

- “Option two,” a number, or an exact option label.
- “Two, actually three” to correct a choice before submitting it.
- “My answer is …” or “Tell Orchestrator …” for a freeform answer.
- “Repeat,” “Explain option two,” “Skip,” “What's pending?” or “On screen.”
- “Cancel that answer” to withdraw an unsent answer or request a correction.

The explicit freeform prefix distinguishes answers from unrelated Aiden requests.
Ambiguous choices require clarification. Audio already playing may not be
recallable; subsequent answers are still checked against current question state.
The speaker firmware suppresses microphone delivery while playing. No new wake
chime or firmware change is included.

## Placement and transport

The Mini pipeline owns the persistent question store and speaker runtime. The
MacBook runs the standalone Claude adapter. Its loopback HTTP client reaches the
Mini control endpoint through a dedicated SSH local forward:

```sh
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:13012:127.0.0.1:3012 infinitewatts@macmini
```

This is a planned activation command, not an instruction to start an unapproved
service. Do not bind either endpoint to a public/LAN address or change SSH access
rules. The adapter disallows environment proxies and redirects. Existing local
processes are within the control endpoint's trust boundary; the microphone is not.

Pipeline configuration, added only for the approved mode:

```yaml
orchestrator_bridge:
  mode: observe
  host_session_id: local_043ec9c0-8316-4e28-8304-5b405c514315
  device: onju-8DAB80
  state_path: /Users/infinitewatts/.local/state/aiden-orchestrator/questions.sqlite
```

`off` omits bridge behavior. `observe` records what would happen without playback
or voice interception. `live` permits the reviewed question/answer scope and
requires the owner's explicit promotion. Do not promote existing observe records
into live questions; ask the originating session to submit a fresh question.

## Persistence and privacy

State is private to the operating-system user. The store keeps question/answer
records needed for correlation and recovery, with bounded pending/total capacity;
full capacity fails closed rather than silently evicting idempotency records.
Observe adapter files retain metadata, not question or answer text. Idle ambient
speech stays under the existing local wake-recognition policy and never enters
this bridge or its history. Do not place secrets in questions, options, or answers.

A local clock supplies expiry. Losing a client lease closes its voice question.
On an uncertain submission or acknowledgement, inspect the existing ID/version
before retrying. Never create a fresh ID merely to retry an uncertain answer.

## Validation and activation

Before activation:

1. Pass core state, parser, adapter, runtime, and existing pipeline regressions.
2. Exercise the prepared adapter via its real stdin/stdout framing and observe
   mode; test private fixture transport independently of the live speaker.
3. Review exact source and configuration, including rollback and health checks.
4. Prepare a private bundle with the adapter, observe configuration, proposed
   hook/MCP fragments, and tunnel command. Do not overwrite Claude settings.
5. Obtain one explicit activation decision for this host and question/answer
   scope. Merge the reviewed fragments preserving existing hooks and MCP servers;
   enroll the intended session and verify its receipt before allowing live use.
6. Test a structured choice, explicit freeform question, desktop handoff, stale
   reply, and unrelated Aiden request. Confirm the same-session receipt, not just
   spoken output or transport success.

Live latency remains a measured acceptance item. The design bypasses Aiden's
paid conversational model for locally resolved replies and keeps a persistent
SSH transport, but it does not promise sub-second audible response before a real
end-to-end benchmark. TTS, silence detection, and Claude hook completion each
contribute. Runtime metadata should separate preparation/playback and receipt
latencies; never use raw ambient transcript logging to measure them.

## Recovery

Retain the previous reviewed Mini SHA and byte-exact configuration before a
pipeline-only deployment. Compare-and-swap configuration updates must preserve
concurrent edits. Automatically restore the previous SHA/configuration and
restart the pipeline when device registration, fresh audio, or expected bridge
mode health checks fail. Preserve the database and receipts for reconciliation.

For a bridge fault after activation, disable the bridge first, restore the exact
prior Claude configuration fragments and owned tunnel job, and verify ordinary
“Hey Aiden” remains functional. Do not terminate unrelated SSH jobs or watchers.
Never roll back the accepted wake-recognition configuration merely to disable
this bridge. Historical pilot wake-watch grants remain revoked.
