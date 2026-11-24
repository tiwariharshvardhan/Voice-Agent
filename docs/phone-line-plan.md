# Phone Line — Change List

> What it takes to put `main.py` on a real phone call instead of a browser tab.
> Written 2026-11-24 against `main.py` @ cc41cf6.

---

## 1. The core insight

The voice pipeline is provider-agnostic already. `connect_sarvam_ws`,
`sarvam_reader`, `run_turn`, `stream_llm_producer`, `_stream_once`,
`tts_consumer`, `flush_sentences`, `TOOLSETS` — **none of this cares where the
audio comes from.**

What breaks is that `index.html` is silently doing **four jobs** that no phone
call will do for you:

| Browser does today | On a phone call |
|---|---|
| VAD → sends `speech_start` / `speech_end` / `speech_cancel` / `interrupt` | nobody sends these |
| Captures 16 kHz PCM via AudioWorklet | carrier gives 8 kHz μ-law |
| Playback queue + `playbackGen` / `dropStaleAudio` barge-in drop | audio is already buffered **at the carrier** |
| `Apply` → `{"type":"settings"}` with persona/voice/tools | no UI, no Apply |

So the work is: **replace the browser, keep the pipeline.**

---

## 2. Architecture diff

Today:
```
Browser ──PCM16k/binary──> /ws ──> Sarvam ──> Groq ──> ElevenLabs ──mp3──> Browser
   └── VAD, playback, barge-in, settings
```

Target:
```
POST /call ──> provider REST ──> dials caller
                                      │ on answer
                          GET /call/answer ──> XML pointing at wss://…/ws/phone
                                      │
Carrier ──μ-law8k/JSON──> /ws/phone ──> [VAD] ──> Sarvam ──> Groq ──> ElevenLabs
                              ▲                                          │
                              └────── PhoneTransport (frame + pace) ─────┘
```

`/ws` stays exactly as it is. The browser demo keeps working.

---

## 3. The changes

### 3.1 Refactor: lift the session out of the `/ws` closure  — *the only structural change*

Right now `run_turn` is a closure inside `websocket_endpoint`, capturing
`conversation_history`, `voice_id`, `active_tools`, `account`, `exec_tool`,
`interrupt_event`, `sarvam_ws`. Two endpoints can't share a closure.

Pull it into a `Session` class holding that state, with `handle_speech_start()`,
`handle_speech_end()`, `handle_interrupt()`, `run_turn()`. Both endpoints then
drive the same object. Mechanical, no behaviour change, and `test_tools.py`
should still pass after it.

### 3.2 `PhoneTransport` — a shim that looks like a `WebSocket`

This is what keeps the pipeline untouched. `run_turn` and `tts_consumer` only
ever call `ws.send_json(...)` and `ws.send_bytes(...)`. Give them an object with
those two methods:

- `send_json(msg)` — `user_transcript`, `agent_transcript`, `tool_call`,
  `ready`, `tts_end`, `tts_fallback` have no meaning on a phone leg → log them.
  `interrupted` → send the provider's **clear/flush** event (see 3.5).
- `send_bytes(audio)` — slice into 20 ms frames, base64, wrap in the provider's
  media envelope, and **pace** the sending. Drop its own queue on interrupt.

Because the shim owns pacing and dropping, `tts_consumer` keeps buffering whole
sentences exactly as it does now. **Zero changes to `run_turn`,
`stream_llm_producer`, `tts_consumer`** beyond 3.4.

### 3.3 Sarvam does the VAD — migrate to the realtime endpoint

**Verified against Sarvam's docs 2026-11-24.** This item was going to be ~40
lines of ported energy VAD. It is now zero.

The realtime endpoint `wss://api.sarvam.ai/speech-to-text-realtime/ws` takes
`endpointing=vad` (the **default**) and emits `vad.speech_start` /
`vad.speech_end` itself. Those map one-to-one onto the `Session` methods that
already exist. Tuning knobs, with Sarvam's defaults against our browser values:

| Sarvam (ms) | default | our browser constant |
|---|---|---|
| `threshold` | 0.3 | dynamic, ~1.05x noise floor |
| `silence_duration_ms` | 1000 | `SILENCE_DELAY_MS` 2200 |
| `min_speech_duration_ms` | 250 | `MIN_SPEECH_MS` 400 |
| `prefix_padding_ms` | 300 | — |

The first three are changeable mid-session via `config.update`;
`prefix_padding_ms` appears to be connection-only (docs don't say outright).

**It is a different protocol, not a flag.** Budget for it:

| | legacy (today) | realtime |
|---|---|---|
| path | `/speech-to-text/ws` | `/speech-to-text-realtime/ws` |
| field name | `type` | `event` |
| audio msg | `{"audio":{"data":…,"sample_rate":…,"encoding":…}}` | `{"event":"audio_input","audio":"<b64>"}` |
| codec param | `input_audio_codec` | `encoding` |
| transcript | `{"type":"data","data":{…}}` | `transcript.partial` / `transcript.final` |
| model | `saaras:v3` | `saaras:v4` (default) |
| auto-detect | **omit** `language-code` | `language_code=auto` |

So `connect_sarvam_ws`, `sarvam_reader` and `Session.feed_audio` all change, and
`on_speech_end`'s flush-then-wait-with-10s-timeout goes away — under vad mode
the transcript arrives as `transcript.final` on Sarvam's schedule, not ours.

**Do it in two stages.** Stage 1 — **DONE** (2026-11-24): browser leg moved to
realtime with `endpointing=manual`, keeping today's client-driven flow. Probed
against the live API with a real key: `session.begin` echoed the config back and
a manual turn (`speech_start` → audio → `speech_end` → `transcript.final`)
completed. Stage 2: the phone leg reuses the same client with `endpointing=vad`,
`encoding=mulaw`, `sample_rate=8000`. One Sarvam client, two configs.

Confirmed while probing: `language` comes back hyphenated (`"en-IN"`), so
`tts_consumer`'s `lang_code.split("-")[0]` keeps working untouched. The session
default is `stream_type=balanced`; a `fast` option exists and is worth measuring
for turn latency, but that's tuning, not migration.

**Latency bug this exposed — FIXED (2026-11-24).** Sarvam returns
`transcript.final` with empty `text` immediately on silence, but `sarvam_reader`
was only queueing non-empty text, so `on_speech_end` burned its full 10 s
timeout before saying "samajh nahi aaya". Now queues empty finals too:
**measured 0.30 s against the live API, down from 10 s.** The 10 s timeout stays
as a backstop for a dropped socket.

This assumes **one `transcript.final` per manual turn** — the same assumption
the queue-clear at `speech_start` has always made. If a live mic test ever loses
a real transcript, suspect Sarvam emitting an interim empty final before the
real one, and look here first.

**Watch for:** `saaras:v3` → `saaras:v4` is a model change and the personas were
tuned on v3. Auto-detect flips from *omitting* the param to passing `auto` —
the opposite of the legacy quirk recorded in the handoff.

### 3.4 Audio format — μ-law needs no conversion

**Inbound: nothing to do.** The realtime endpoint accepts `encoding=mulaw`
(enum: `linear16`, `linear32`, `mulaw`, `alaw`) at `sample_rate=8000`. Carrier
bytes go straight to Sarvam. No `audioop-lts`, no lookup table, no resampling —
scratch the Python 3.13 problem entirely. Mono only; frames capped at 16,000
bytes, and ~100 ms chunks are what the guide suggests.

**Outbound:** ElevenLabs emits `ulaw_8000` via the `output_format` query param.
One new argument on `tts_consumer`, since the browser path stays MP3.

### 3.5 Barge-in must tell the carrier to shut up

The one people get wrong. Today an interrupt stops playback *in the browser*. On
a call, several seconds of audio are already sitting in the carrier's jitter
buffer — setting `interrupt_event` does nothing to it, and the caller keeps
hearing the old sentence over their own voice.

So on interrupt, the transport must send the provider's clear command
(Twilio: `{"event":"clear"}`; Plivo: `clearAudio`) **and** drop its own pending
frame queue. This is what 3.2's `interrupted` mapping is for.

### 3.6 The agent has to speak first

Every turn today is reactive: `speech_end` → `run_turn`. On an outbound call the
**agent** opens ("Hello, this is Asha from ConnectTel…"). There is currently no
code path for an agent-initiated turn — `run_turn` requires a user transcript.

Needs a greeting path that pushes a first utterance into the TTS queue on
connect, before any VAD event. Small, but nothing today does it.

### 3.7 Persona selection without an `Apply` button

`{"type":"settings"}` is how the toolset, voice, prompt and language get chosen,
and the handoff notes it's **per-connection** — required again after every
refresh. A phone call can't send it.

Decide: hardcode the phone leg to `telecom`, or accept persona in the `POST
/call` body and stash it per-call keyed on the provider's call id, to be read
when the WS arrives. Latter is barely more work and keeps the demo flexible.

### 3.8 Call lifecycle

- **Hangup** — provider `stop` event or WS close → cancel `active_turn`, close
  Sarvam, free the session.
- **History is unbounded.** Accepted today because a page refresh resets it. A
  phone call has no refresh, so a long call grows context until it gets
  expensive or 400s. Needs a cap now, not later.
- **Call duration cap** — hard limit, or a stuck call bills forever.

### 3.9 Two new HTTP endpoints

- `POST /call` — takes a number, calls the provider's REST API.
- `GET /call/answer` — returns the XML that points the carrier at
  `wss://…/ws/phone`.

Plus provider credentials in `.env`, and `requirements.txt` additions.

### 3.10 Abuse controls — not optional

A public "type any number, get a call" endpoint is a free robodialer for anyone
who finds it, billed to you.

- **OTP-verify the number before dialing** (also your consent record)
- per-IP and per-number rate limit
- global daily spend cap + max concurrent calls
- never dial a number that didn't just pass OTP

This is a trust boundary. ~30 lines, and it gates 3.9 going public.

### 3.11 Frontend: a "call me" form

Phone input → OTP input → call status. New UI in `index.html`, independent of
everything above.

### 3.12 Reachability

The carrier must reach your server over the public internet. Railway already
gives you that; local testing needs ngrok or similar.

---

## 4. What does NOT change

`banking_tools.py`, `telecom_tools.py`, `TOOLSETS`, `flush_sentences`,
`accumulate_tool_calls`, `normalize_args`, `salvage_failed_tool_call`,
`strip_fn_tags`, `MAX_HOPS` hop-loop, the Groq body, the `/ws` endpoint, the
browser demo, `test_tools.py`, `eval_tools.py`.

---

## 5. Open decisions (block coding)

1. **Which provider** — the envelope format, the clear command, the REST call
   and the XML dialect are all provider-specific. "voicelink" was mentioned;
   I don't know it — needs its docs. Everything in §3 is provider-shaped except
   3.1–3.3, which can start now.
2. **Inbound or outbound** — inbound (you publish a number) deletes 3.6, 3.9,
   3.10 and 3.11 outright. Outbound is the more impressive demo and carries all
   of them.
3. ~~Does Sarvam do endpointing~~ — **answered: yes.** See 3.3.

---

## 6. Risks worth knowing

- **Narrowband hurts.** 8 kHz μ-law will make the ElevenLabs voice noticeably
  worse than the browser demo. That's the phone network, not a bug — but your
  demo's "wow" partly rests on voice quality.
- **Latency goes up.** Carrier adds ~100–200 ms round trip on top of the
  per-utterance Sarvam reconnect. The `<2s` TTFA target gets harder, and the
  latency badge has no screen to live on any more.
- **Per-turn Sarvam reconnect** goes away with 3.3. The realtime endpoint
  holds the socket open and only drops after 60 s of silence (close 1008,
  `inactivity_timeout`), so send `{"event":"ping"}` on idle. Today's
  open-a-socket-per-utterance cost disappears.
- **Concurrency.** Sessions are isolated (`account` is a per-connection
  deepcopy), but Groq / ElevenLabs / Sarvam rate limits and your Railway
  instance size are now shared across simultaneous callers.
