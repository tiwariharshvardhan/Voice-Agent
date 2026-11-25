# Phone Line — Change List

> What it takes to put `main.py` on a real phone call instead of a browser tab.
> Written 2026-11-24 against `main.py` @ cc41cf6.

---

## 0. Status (2026-11-25)

**The inbound leg is built and works end to end**, verified without a phone by
synthesizing A-law speech with ElevenLabs and playing it into `/ws/phone` as
VoiceLink would.

Built: `PhoneTransport`, `POST /voicelink/inbound`, `POST /voicelink/webhook`,
`WS /ws/phone`, VAD-driven turns, agent-speaks-first greeting, A-law TTS output,
and `telecom_tools.SYSTEM_PROMPT` (the persona, server-side at last).

Measured on that test:

| | |
|---|---|
| greeting, time to first audio | **0.96 s** |
| reply, after the caller stops speaking | **2.78 s** |
| full turn | transcript → `get_account` → spoken answer, correct |

The 2.78 s includes Sarvam's `silence_duration_ms` (default **1000 ms**) spent
waiting to be sure the caller finished — so ~1.8 s is actual work. That knob is
the cheapest latency lever available and is changeable mid-session.

**Not built:** outbound (`add_lead`), abuse controls (3.10), the call-me form
(3.11). **Not possible yet:** a real call — needs KYC and a DID.

### Found while testing, worth fixing

1. **gpt-oss invents arguments for no-arg tools.** Saw
   `get_account({"n":1})` → `error: invalid tool or arguments`, then a retry
   with `{}`. The hop-loop self-corrects, but it costs a round trip mid-call.
   Intermittent (1 of 2 runs). Making dispatch ignore unknown kwargs kills it.
2. **Money read wrong.** `₹12.5` was spoken as "do point five rupees" — the 12
   became "do". A prompt or tool-format issue, and it's the single worst class
   of error for a telecom bot.
3. **Dates leak their punctuation.** `2026-08-14` became "twenty twenty six
   dash fourteen August". The tool returns ISO dates and the prompt's
   no-special-characters rule does not survive the tool result.
4. **STT hallucinated on synthetic silence** — appended "साया हमारा नासोहा।" to
   a correct transcript. Probably an artifact of feeding a constant A-law
   silence byte; real carrier lines carry comfort noise. Watch for it on a real
   call before treating it as a bug.

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

VoiceLink's protocol is Twilio Media Streams in all but name, so the shim is
small. `run_turn` and `tts_consumer` only ever call `send_json` / `send_bytes`.

**VoiceLink → us** (JSON text frames):
```
{"event":"connected"}
{"event":"start","sequence_number":0,"stream_sid":"stream_abc123",
 "start":{"stream_sid":"…","call_sid":"…","account_sid":"42",
          "from":"+14155550100","to":"+14155550199",
          "custom_parameters":{…},
          "media_format":{"encoding":"audio/alaw","sample_rate":"8000"}}}
{"event":"media","media":{"track":"inbound","payload":"<base64>"}}
{"event":"stop","stop":{"callSid":"callabcdef1234"}}
```

**us → VoiceLink:**
```
{"event":"media","media":{"payload":"<base64>"}}
{"event":"clear","stream_sid":"<stream sid>"}
{"event":"mark","mark":{"name":"greeting_done"}}
{"event":"transfer","target":"1234567890"}
```

So `send_json` maps the browser protocol onto this: `interrupted` → `clear`,
everything else (`user_transcript`, `tool_call`, `ready`, `tts_end`) → a log
line. `send_bytes` base64s and wraps in a `media` event, in segments **under
2 seconds** (VoiceLink's documented guidance; exact chunk size is undocumented).

**Mind the casing.** `start` gives `stream_sid` / `call_sid`; `stop` gives
`callSid`; `clear` wants `stream_sid`. Read both spellings, don't assume.

**Bonus — `transfer` makes a tool real.** `telecom_tools.transfer_to_human`
currently only *says* it's transferring. With `{"event":"transfer","target":…}`
it can actually hand the call to a mobile number. Given the whole ConnectTel
story is about never reaching a human, that's the strongest beat in the demo.

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

### 3.4 Audio format — A-law, and nothing to convert

**Correction to the earlier plan: VoiceLink is A-law, not μ-law.** The `start`
event says `media_format.encoding = "audio/alaw"`, `sample_rate = "8000"`.
(Their overview page also says "raw SLIN" and "PCM" in places — the docs
contradict themselves, so this needs confirming with VoiceLink.)

Taking A-law as correct, **both ends still need zero DSP** — verified live
2026-11-24:

| | result |
|---|---|
| Sarvam `encoding=alaw`, `sample_rate=8000` | `session.begin` echoed `encoding=alaw rate=8000` |
| ElevenLabs `output_format=alaw_8000` | HTTP 200, 2972 bytes |

So carrier A-law goes straight into Sarvam, and ElevenLabs emits A-law straight
back out. No `audioop-lts`, no lookup table, no resampling — the Python 3.13
problem never materialises. `tts_consumer` needs one `output_format` argument,
since the browser path stays MP3.

### 3.5 Barge-in — `clear`, and it is documented

Confirmed: `{"event":"clear","stream_sid":"<sid>"}` "discards queued audio not
yet played". That is exactly the carrier-side flush this item needed, so the
transport's `interrupted` mapping has a real target.

No acknowledgement event is documented, so treat it as fire-and-forget and drop
our own pending frame queue at the same time.

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

### 3.9 The VoiceLink API — outbound is a lead queue, not a dial

This is the biggest surprise and it changes 3.10 and 3.11.

**Auth is a login, not a static key.**
`POST https://app.voicelink.co.in/api/v1/auth/login` with `username` /
`password` returns `data.access_token` (a Laravel Sanctum token), then
`Authorization: Bearer {access_token}`. So there's token acquisition and
refresh to handle — more than dropping a key in `.env`.

**Outbound: `POST /api/v1/add_lead`.**
Params: `did_number` (mandatory), `customer_number` (**10 digits, no `91` or
`+91` prefix**), `country_code: "91"`, optional `leads[]` array,
`websocket_url` and `webhook_url` as **per-call overrides**, `call_limit`.

It responds `"Leads queued successfully"` with `bot_id` / `client_id` /
`accepted_leads` — **and no call SID.** This is a campaign queue, not a
request-response dial. To find the resulting call:
`GET /api/v1/call-log/details?call_id=…`.

**What that means for "visitor clicks, gets a call":** we can't hand the user a
call id synchronously, and dial timing is the queue's, not ours. The web UI has
to be fire-and-forget plus webhook-driven status, not a spinner waiting on a
call object.

**The `websocket_url` per-call override is the useful part** — it carries the
persona choice (3.7) as a query param, so no server-side call-id registry.

**Inbound is cleaner than outbound.** VoiceLink POSTs per call to a URL of ours
with `callerId`, `customerNumber`, `callId`, `sid`, and we reply with JSON
`ws_url` + `webhook_url` (both required). If `ws_url` drops it retries **once**,
then the call ends `failed`. Dynamic routing per call, no dashboard round-trip.

**Webhooks:** `call.initiated` / `call.answered` / `call.ended` /
`call.completed`, carrying `callId`, `fromNumber`, `toNumber`, `direction`,
`callStatus`, `duration`, `recordingUrl`, `customParameters`. Two different
payload schemas are documented for outbound vs inbound — confirm which applies.

**`recordingUrl` is a free win.** VoiceLink records calls, so the demo
recordings that were going to be a manual chore arrive as webhook payloads.

**KYC is real but a student can clear it.** Individual: PAN (name-matched) +
Aadhaar via DigiLocker. GST only for business accounts. DLT is documented only
for 140/160-series DID purchases (`dlt_confirmed`, `use_case`), not as a blanket
outbound requirement — but nothing in the docs covers TRAI rules on automated
calling, so don't read silence as permission.

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

## 5. Open decisions

1. ~~Which provider~~ — **VoiceLink.** Wire format documented and read
   (2026-11-24); see 3.2, 3.4, 3.5, 3.9.
2. ~~Inbound or outbound~~ — **inbound, and it is built** (see §0). Note the
   plan was wrong that inbound deletes 3.6: a caller who dials and hears
   silence is worse than one who is greeted, so the greeting was needed anyway.
   Outbound remains open and is queue-shaped, not instant.
3. ~~Does Sarvam do endpointing~~ — **answered: yes.** See 3.3.
4. **Account state** — is KYC done, is there a DID, is there a trial
   restriction? Nothing is testable end to end until a number exists.

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

---

## 7. Ask VoiceLink before building against this

The docs contradict themselves in places and the full API text file was
truncated when read, so these are assumptions, not facts:

1. **Audio frame format** — overview says "raw SLIN"/PCM, the `start` event
   says `audio/alaw`. Which is it, and is it always base64-in-JSON (never
   binary frames)?
2. **Chunk size** — only "under 2 seconds" is documented. What is the actual
   cap per `media` frame?
3. **Which webhook schema** applies to outbound vs inbound.
4. **`add_lead` response** — is there any call identifier returned, and what is
   the field called?
5. **Concurrency and CPS** — `channel_count` and `call_limit` exist; no maximum
   is published.
6. **Trial limits and TRAI/DLT for ordinary outbound** — are there verified-
   number restrictions, and is DLT needed outside the 140/160 series?

Pages not read yet: the integrations list (Pipecat / LiveKit may document the
frame format more precisely), plans, the CPS calculator, and the 140/160
registration process.
