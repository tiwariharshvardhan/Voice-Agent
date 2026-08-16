# Voice Agent

A low-latency, multilingual, interruptible voice agent with real tool calling - built to demonstrate the difference between a voice agent and a voice chatbot.

**Live demo:** https://voice-agent-phi-six.vercel.app/

Speak to it in English, Hindi, Hinglish, or 8 other Indian languages. Interrupt it mid-sentence. Ask the banking or telecom persona to actually do things - check a balance, block a card, register a complaint - and watch the tool calls happen in the transcript.

## Features

- **Streaming speech-to-text** - Sarvam AI Saaras v3 over WebSocket, 80-190 ms processing latency, automatic language detection across 11 Indian languages
- **Barge-in** - interrupt the agent while it speaks; the server rolls back the partial turn (including tool messages) and the client discards stale audio
- **Tool calling** - banking (3 tools) and telecom (7 tools) personas execute real functions through a streaming hop loop that self-corrects on malformed arguments
- **Sentence-level TTS streaming** - ElevenLabs Flash v2.5 synthesizes each sentence as the LLM streams it, so speech starts before the reply is finished
- **Language-aware pronunciation** - the language detected by STT is passed to TTS, so Hindi replies are spoken with Hindi pronunciation, not English guesses
- **Five personas** - e-commerce, healthcare, banking, food delivery, and telecom customer care, each with its own system prompt and voice
- **Latency badge** - every agent message shows time-to-first-audio, the metric that actually matters in voice UX

## Architecture

One conversation turn:

```
Mic
 └─ AudioWorklet (16 kHz PCM, 40 ms chunks)
     └─ Browser VAD (AnalyserNode, calibrated noise floor)
         ├─ speech_start -> backend opens Sarvam WS (per-utterance)
         ├─ PCM chunks -> base64 -> Sarvam
         └─ speech_end  -> flush -> transcript + detected language
             └─ Groq Llama 3.3 70B (streaming, tool calling, max 4 hops)
                 └─ sentences -> ElevenLabs Flash v2.5 (per-sentence)
                     └─ binary audio -> browser playback queue

Barge-in: VAD keeps running during playback. On sustained speech the client
interrupts, the server truncates the turn, and stale audio chunks are dropped
via a playback generation counter.
```

Design choices worth knowing:

- **Per-utterance STT connections.** Sarvam closes idle WebSockets within seconds, so a connection is opened on speech start rather than held open. Config travels in URL query parameters, not a JSON handshake.
- **Tool-call salvage.** Llama 3.3 intermittently emits tool calls as malformed raw text, which Groq rejects with a 400. The server parses the intended call out of the error's `failed_generation` and executes it anyway, so the turn never goes silent.
- **Interrupt rollback is turn-scoped.** A tool turn appends multiple messages to history; on interrupt the whole turn is sliced off so history never ends on a dangling tool message.
- **Per-connection state.** Each WebSocket session gets a deep copy of the demo account data - sessions never share state.

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + WebSockets (Python) |
| Frontend | Single-page `index.html`, no framework, no build step |
| STT | Sarvam AI Saaras v3 (WebSocket streaming) |
| LLM | Groq - Llama 3.3 70B (streaming + tool calling) |
| TTS | ElevenLabs Flash v2.5 (per-sentence streaming) |

## Performance

| Metric | Value |
|---|---|
| STT processing latency | 80-190 ms |
| Tool-call accuracy (20-utterance eval, EN/HI/Hinglish) | 95% |
| Time-to-first-audio target | < 2 s (shown per message in the UI) |

## Project structure

```
main.py               FastAPI app: WebSocket endpoint, Sarvam client, LLM hop
                      loop, sentence flushing, TTS consumer, TOOLSETS registry
index.html            Full UI and client logic: AudioWorklet capture, VAD,
                      barge-in, playback queue, personas and prompts
banking_tools.py      get_balance, recent_transactions, block_card
telecom_tools.py      account, charges, outages, number status, complaints,
                      ticket status, human handoff
test_tools.py         Assert-based self-check (no framework)
eval_tools.py         Tool-call accuracy eval against Groq
docs/                 Telecom agent design: journey map, intent-tool matrix,
                      knowledge base, demo call scripts
```

## Getting started

Requires Python 3.11+ and API keys for Sarvam AI, Groq, and ElevenLabs.

```bash
git clone https://github.com/tiwariharshvardhan/Voice-Agent.git
cd voice-agent
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
SARVAM_API_KEY=your_sarvam_key
GROQ_API_KEY=your_groq_key
ELEVENLABS_API_KEY=your_elevenlabs_key

# Optional
ELEVENLABS_VOICE_ID=your_preferred_voice_id
ELEVENLABS_MODEL=eleven_flash_v2_5   # eleven_turbo_v2_5 for higher quality
SARVAM_DEBUG=1                       # log raw STT wire traffic
```

Run:

```bash
uvicorn main:app --reload --port 8000
```

Open http://127.0.0.1:8000, allow microphone access, open Settings, pick a persona, press Apply, then Start. Settings apply per connection - press Apply again after any page refresh.

## Testing

```bash
python3 test_tools.py    # tool functions, dispatch errors, streaming
                         # reassembly, arg normalization, salvage parsing
python3 eval_tools.py    # tool-call accuracy across EN/HI/Hinglish utterances
```

## Adding a new toolset

1. Create `yourdomain_tools.py` exposing a tool schema list, a `new_account()` factory, and an `execute_tool(name, args_json, account)` dispatcher that returns error strings instead of raising.
2. Register it in the `TOOLSETS` dict in `main.py`.
3. Add a persona prompt in `index.html`.

Unknown toolset names degrade gracefully to a no-tools conversation.

## Notes

- The demo accounts, plans, and tickets are fictional. The telecom persona ("ConnectTel") is a generic fictional operator.
- The agent always identifies itself as an AI and every telecom call ends in exactly one of three states: resolved, ticket registered, or transferred to a human.
- Browser support: tested on desktop Chrome and mobile Safari. Barge-in depends on browser echo cancellation, so headphones give the best experience.
