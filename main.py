import asyncio
import base64
import os
import json
import re
import ssl
from urllib.parse import urlencode
import certifi
import httpx
from websockets.asyncio.client import connect as ws_connect

SSL_CTX = ssl.create_default_context(cafile=certifi.where())
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

import banking_tools
import telecom_tools

# toolset name -> (tool schemas, per-connection account factory, dispatcher)
TOOLSETS = {
    "banking": (banking_tools.BANKING_TOOLS, banking_tools.new_account, banking_tools.execute_tool),
    "telecom": (telecom_tools.TELECOM_TOOLS, telecom_tools.new_account, telecom_tools.execute_tool),
}

load_dotenv()

app = FastAPI()

SARVAM_API_KEY      = os.getenv("SARVAM_API_KEY")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "bajNon13EdhNMndG3z05")
ELEVENLABS_MODEL    = os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")
# language_code enforcement only exists on these models; others 400 on the param
ELEVENLABS_LANG_MODELS = {"eleven_flash_v2_5", "eleven_turbo_v2_5"}

SARVAM_STT_WS = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
SARVAM_MODEL  = os.getenv("SARVAM_MODEL", "saaras:v4")   # saaras:v3-realtime to fall back
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")   # https origin VoiceLink reaches us on

DEBUG = os.getenv("SARVAM_DEBUG") == "1"   # set SARVAM_DEBUG=1 to log the full Sarvam wire traffic

SYSTEM_PROMPT = (
    "You are a helpful and friendly voice assistant on a live voice call. "
    "Respond in the same language the user speaks. "
    "Keep every reply to at most 3 spoken sentences — your words are read aloud, "
    "so a long answer means the caller waits and listens for a minute. "
    "Be specific, not vague: give actual numbers, prices, and names, and never hedge "
    "with 'it depends' unless you truly cannot estimate. "
    "For broad questions (like planning a trip), do NOT list everything at once — "
    "give the single most useful piece of information, then ask one follow-up question "
    "to narrow down what the caller needs. Speak naturally, like a friend on the phone."
)

SENTENCE_ENDERS = frozenset('.!?।\n')
MIN_SENTENCE_LEN = 15


def strip_fn_tags(text: str) -> str:
    # Llama 3.3 intermittently emits tool calls as raw text; tags can arrive
    # split across stream tokens, so strip on assembled sentences, not deltas.
    text = re.sub(r"<function=\w+>.*?</function>", "", text, flags=re.DOTALL)
    return re.sub(r"</?function[=\w]*>", "", text)


def flush_sentences(buffer: str) -> tuple[list[str], str]:
    sentences = []
    while True:
        found = -1
        for i, ch in enumerate(buffer):
            if ch in SENTENCE_ENDERS and i >= MIN_SENTENCE_LEN - 1:
                found = i
                break
        if found == -1:
            break
        sentence = buffer[:found + 1].strip()
        if sentence:
            sentences.append(sentence)
        buffer = buffer[found + 1:].lstrip()
    return sentences, buffer


async def connect_sarvam_ws(language_code: str, endpointing: str = "manual",
                            encoding: str = "linear16", sample_rate: int = 16000):
    """Open a Sarvam realtime STT WebSocket. Config goes in URL params.

    endpointing="manual" — we delimit turns with speech_start/speech_end.
    endpointing="vad"    — Sarvam finds turn boundaries and emits vad.speech_*.
    """
    params = {
        "model": SARVAM_MODEL,
        "mode": "transcribe",
        "endpointing": endpointing,
        "encoding": encoding,
        "sample_rate": str(sample_rate),
        # realtime wants an explicit "auto"; the legacy WS required omitting it
        "language_code": language_code if language_code and language_code != "unknown" else "auto",
    }
    url = f"{SARVAM_STT_WS}?{urlencode(params)}"
    if DEBUG: print(f"Connecting to Sarvam: {url}")
    return await ws_connect(url, additional_headers={"api-subscription-key": SARVAM_API_KEY}, ssl=SSL_CTX)


async def sarvam_reader(sarvam_ws, transcript_queue: asyncio.Queue, on_vad=None):
    """Drain Sarvam responses, push transcripts to the queue. `on_vad` is awaited
    with "vad.speech_start"/"vad.speech_end" — only sent when endpointing=vad."""
    try:
        async for msg in sarvam_ws:
            data = json.loads(msg)
            if DEBUG: print(f"Sarvam msg: {data}")
            event = data.get("event")
            if event == "transcript.final":
                # Queue it even when `text` is empty: an empty final is Sarvam
                # saying "nothing there", and dropping it left on_speech_end
                # waiting out its 10s timeout — 10s of dead air on a mishear.
                # Assumes one final per manual turn, which is what the queue
                # being cleared at speech_start has always assumed.
                # `language` is only present when language_code=auto
                await transcript_queue.put((data.get("text", ""), data.get("language", "")))
            elif event in ("vad.speech_start", "vad.speech_end"):
                if on_vad is not None:
                    await on_vad(event)
            elif event == "error":
                print(f"Sarvam error [{data.get('code')}] fatal={data.get('is_fatal')}: {data.get('message')}")
    except Exception as e:
        print(f"Sarvam reader closed: {e}")


@app.get("/")
async def get():
    return HTMLResponse(open("index.html").read())


def _on_turn_done(task) -> None:
    try:
        task.result()
    except (asyncio.CancelledError, Exception):
        pass


class Session:
    """One voice conversation. `transport` is anything exposing send_json /
    send_bytes — a browser WebSocket today, a phone-leg shim later."""

    def __init__(self, transport):
        self.transport        = transport
        self.system_prompt    = SYSTEM_PROMPT
        self.history          = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.voice_id         = ELEVENLABS_VOICE_ID
        self.tts_format       = ""        # "" = ElevenLabs default (mp3, browser); "alaw_8000" on a call
        self.language_code    = "unknown"
        self.active_tools     = None
        self.account          = None      # per-connection — sessions never share
        self.exec_tool        = None
        self.active_turn      = None
        self.interrupt_event  = asyncio.Event()
        self.transcript_queue = asyncio.Queue()
        # Sarvam WS opened per-utterance on speech_start (avoids idle timeout)
        self.sarvam_ws        = None
        self.reader_task      = None

    async def apply_settings(self, data: dict) -> None:
        self.language_code = data.get("language_code", self.language_code)
        self.voice_id      = data.get("voice_id", self.voice_id)
        self.system_prompt = data.get("system_prompt", self.system_prompt)
        toolset = TOOLSETS.get(data.get("tools"))
        self.active_tools, self.account, self.exec_tool = (
            (toolset[0], toolset[1](), toolset[2]) if toolset else (None, None, None))
        self.history = [{"role": "system", "content": self.system_prompt}]
        print(f"Settings updated — voice: {self.voice_id}, lang: {self.language_code}, tools: {data.get('tools') or 'none'}")
        await self.transport.send_json({"type": "settings_ack"})

    async def open_stt(self, on_vad=None, **kw) -> bool:
        """Open the Sarvam socket and start its reader. False on failure — STT
        being unreachable must never kill the session."""
        try:
            self.sarvam_ws = await connect_sarvam_ws(self.language_code, **kw)
        except Exception as e:
            print(f"Sarvam connect failed: {e}")
            self.sarvam_ws, self.reader_task = None, None
            return False
        self.reader_task = asyncio.create_task(
            sarvam_reader(self.sarvam_ws, self.transcript_queue, on_vad))
        return True

    async def on_speech_start(self) -> None:
        print("STT stream starting")
        while not self.transcript_queue.empty():
            self.transcript_queue.get_nowait()
        if await self.open_stt():
            await self.sarvam_ws.send(json.dumps({"event": "speech_start"}))
        else:
            await self.transport.send_json({"type": "error", "text": "Awaaz seva mein dikkat hai, dobara bolo."})
            await self.transport.send_json({"type": "ready"})

    async def on_speech_end(self) -> None:
        if self.sarvam_ws is None:   # connect failed at speech_start
            await self.transport.send_json({"type": "ready"})
            return
        print("STT stream ending — closing turn")
        await self.sarvam_ws.send(json.dumps({"event": "speech_end"}))

        try:
            transcript, detected_lang = await asyncio.wait_for(self.transcript_queue.get(), timeout=10.0)
        except asyncio.TimeoutError:
            transcript, detected_lang = "", ""

        self.reader_task.cancel()
        await self.sarvam_ws.close()
        self.sarvam_ws = None

        if not transcript:
            await self.transport.send_json({"type": "error", "text": "Samajh nahi aaya, dobara bolo."})
            await self.transport.send_json({"type": "ready"})
            return

        await self.start_turn(transcript, detected_lang)

    async def on_speech_cancel(self) -> None:
        print("STT cancelled (too short)")
        if self.reader_task:
            self.reader_task.cancel()
        if self.sarvam_ws:
            await self.sarvam_ws.close()
            self.sarvam_ws = None
        while not self.transcript_queue.empty():
            self.transcript_queue.get_nowait()
        await self.transport.send_json({"type": "ready"})

    async def on_interrupt(self) -> None:
        print("Interrupt received")
        if self.active_turn and not self.active_turn.done():
            self.interrupt_event.set()

    async def feed_audio_b64(self, payload: str) -> None:
        """Carrier audio arrives already base64'd — don't round-trip it."""
        if not payload or self.sarvam_ws is None:
            return
        await self.sarvam_ws.send(json.dumps({"event": "audio_input", "audio": payload}))

    async def feed_audio(self, pcm: bytes) -> None:
        if not pcm:
            return
        await self.feed_audio_b64(base64.b64encode(pcm).decode())

    async def greet(self) -> None:
        """Agent speaks first, so a caller hears a greeting instead of silence.
        The persona prompt already says to introduce itself."""
        await self.start_turn("", greeting=True)

    async def start_turn(self, transcript: str, tts_lang: str = "", greeting: bool = False) -> None:
        """Cancel any in-flight turn, then run this one in the background."""
        if self.active_turn and not self.active_turn.done():
            self.interrupt_event.set()
            self.active_turn.cancel()
            try: await self.active_turn
            except (asyncio.CancelledError, Exception): pass

        self.interrupt_event.clear()
        self.active_turn = asyncio.create_task(self.run_turn(transcript, tts_lang, greeting))
        self.active_turn.add_done_callback(_on_turn_done)

    async def run_turn(self, transcript: str, tts_lang: str = "", greeting: bool = False) -> None:
        if not greeting and not transcript.strip():
            await self.transport.send_json({"type": "error", "text": "Samajh nahi aaya, dobara bolo."})
            await self.transport.send_json({"type": "ready"})
            return

        turn_start = len(self.history)
        if not greeting:
            print(f"[USER] {transcript}")
            await self.transport.send_json({"type": "user_transcript", "text": transcript})
            self.history.append({"role": "user", "content": transcript})

        sentence_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
        results = await asyncio.gather(
            stream_llm_producer(self.history, sentence_queue, self.interrupt_event,
                                self.active_tools, self.account, self.transport, self.exec_tool),
            tts_consumer(sentence_queue, self.transport, self.voice_id, self.interrupt_event,
                         tts_lang, self.tts_format),
            return_exceptions=True,
        )
        full_text  = results[0] if isinstance(results[0], str) else ""
        audio_sent = results[1] if isinstance(results[1], bool) else False

        if self.interrupt_event.is_set():
            print(f"[INTERRUPTED] {full_text[:60]}")
            # Drop the whole turn — including any tool-call/tool messages the
            # hop-loop appended — so history never ends on a dangling tool msg.
            del self.history[turn_start:]
            await self.transport.send_json({"type": "interrupted"})
            return

        if full_text:
            self.history.append({"role": "assistant", "content": full_text})
        print(f"[AGENT] {full_text}")
        await self.transport.send_json({"type": "agent_transcript", "text": full_text})

        if audio_sent:
            await self.transport.send_json({"type": "tts_end"})
        else:
            await self.transport.send_json({"type": "tts_fallback", "text": full_text})

    async def close(self) -> None:
        if self.active_turn and not self.active_turn.done():
            self.active_turn.cancel()
        if self.reader_task:
            self.reader_task.cancel()
        if self.sarvam_ws:
            try:
                await self.sarvam_ws.close()
            except Exception:
                pass


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected")
    session = Session(ws)

    try:
        while True:
            msg = await ws.receive()

            if "text" in msg:
                data = json.loads(msg["text"])
                msg_type = data.get("type")

                if msg_type == "settings":
                    await session.apply_settings(data)
                elif msg_type == "speech_start":
                    await session.on_speech_start()
                elif msg_type == "speech_end":
                    await session.on_speech_end()
                elif msg_type == "speech_cancel":
                    await session.on_speech_cancel()
                elif msg_type == "interrupt":
                    await session.on_interrupt()
                continue

            if "bytes" in msg and msg["bytes"]:
                await session.feed_audio(msg["bytes"])

    except WebSocketDisconnect:
        print("Client disconnected")
    except RuntimeError as e:
        if "disconnect" in str(e).lower():
            print("Client disconnected mid-session")
        else:
            print(f"Runtime error: {e}")
    except Exception as e:
        print(f"Unhandled error: {e}")
        try:
            await ws.send_json({"type": "error", "text": str(e)})
        except Exception:
            pass
    finally:
        await session.close()



MAX_HOPS = 4  # ponytail: fixed hop cap; raise only if a real multi-tool flow needs it


def normalize_args(args: str) -> str:
    # Llama 3.3 emits literal `null` (or nothing) for no-arg tools; both mean {}.
    # Without this, json.loads gives None -> **None throws -> wasted error hop.
    return "{}" if args.strip() in ("", "null") else args


def salvage_failed_tool_call(err_body: str):
    """Groq 400s with code tool_use_failed when Llama emits a malformed raw-text
    tool call (e.g. `<function=recent_transactions[]{"n": "3"}</function>`).
    Recover name+args from the error's failed_generation so the turn proceeds
    instead of going silent. Returns a call dict or None."""
    try:
        failed = json.loads(err_body)["error"]["failed_generation"]
    except Exception:
        return None
    m = re.search(r"<function=(\w+)[^{<]*(\{.*?\})?\s*(?:</function>|$)", failed, re.DOTALL)
    if not m:
        return None
    return {"id": "salvaged_0", "name": m.group(1), "args": normalize_args(m.group(2) or "")}


def accumulate_tool_calls(acc: dict, deltas: list) -> None:
    """Merge one chunk's tool_call deltas into acc {index: {id, name, args}}.
    Name arrives whole; arguments arrive fragmented across chunks."""
    for tc in deltas:
        slot = acc.setdefault(tc["index"], {"id": "", "name": "", "args": ""})
        if tc.get("id"):
            slot["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name"):
            slot["name"] += fn["name"]
        if fn.get("arguments"):
            slot["args"] += fn["arguments"]


async def _stream_once(history, queue, interrupt, tools):
    """One Groq streaming pass. Text streams to the TTS queue as today;
    tool-call deltas are reassembled. Never emits the None sentinel."""
    text = ""
    buffer = ""
    calls: dict = {}
    body = {
        "model": "openai/gpt-oss-120b",
        "messages": history,
        "max_tokens": 600,
        "stream": True,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=body,
        ) as resp:
            if resp.status_code != 200:
                err = (await resp.aread()).decode(errors="replace")
                call = salvage_failed_tool_call(err)
                if call:
                    print(f"[SALVAGED] raw tool call from Groq {resp.status_code}: {call['name']}({call['args']})")
                    return "", [call]
                raise RuntimeError(f"Groq {resp.status_code}: {err[:300]}")
            async for line in resp.aiter_lines():
                if interrupt.is_set():
                    break
                if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                    continue
                try:
                    delta = json.loads(line[6:])["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta.get("tool_calls"):
                    accumulate_tool_calls(calls, delta["tool_calls"])
                content = delta.get("content") or ""
                if not content:
                    continue
                text += content
                buffer += content
                sentences, buffer = flush_sentences(buffer)
                for s in sentences:
                    if interrupt.is_set():
                        break
                    s = strip_fn_tags(s)
                    if s.strip():
                        await queue.put(s)
    tail = strip_fn_tags(buffer).strip()
    if tail and not interrupt.is_set():
        await queue.put(tail)
    for tc in calls.values():
        tc["args"] = normalize_args(tc["args"])
    return strip_fn_tags(text), [calls[i] for i in sorted(calls)]


async def stream_llm_producer(history: list, queue: asyncio.Queue, interrupt: asyncio.Event,
                              tools=None, account=None, transport=None, exec_tool=None) -> str:
    full_text = ""
    try:
        for _ in range(MAX_HOPS):
            text, tool_calls = await _stream_once(history, queue, interrupt, tools)
            full_text += text
            if not tool_calls or interrupt.is_set():
                break
            history.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["args"] or "{}"}}
                for tc in tool_calls
            ]})
            for tc in tool_calls:
                result = exec_tool(tc["name"], tc["args"], account)
                print(f"[TOOL] {tc['name']}({tc['args']}) -> {result}")
                if transport is not None:
                    await transport.send_json({"type": "tool_call", "name": tc["name"], "result": result})
                history.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"LLM stream error: {e}")
    finally:
        await queue.put(None)
    return full_text


async def tts_consumer(queue: asyncio.Queue, transport, vid: str, interrupt: asyncio.Event,
                       lang_code: str = "", output_format: str = "") -> bool:
    vid = vid or ELEVENLABS_VOICE_ID
    tts_body = {
        "model_id": ELEVENLABS_MODEL,
        "voice_settings": {"stability": 0.7, "similarity_boost": 0.75},
    }
    if lang_code and ELEVENLABS_MODEL in ELEVENLABS_LANG_MODELS:
        tts_body["language_code"] = lang_code.split("-")[0]   # Sarvam "hi-IN" → ISO 639-1 "hi"
    audio_sent = False
    while True:
        sentence = await queue.get()
        if sentence is None:
            break
        if interrupt.is_set():
            while True:
                try:
                    item = queue.get_nowait()
                    if item is None:
                        break
                except asyncio.QueueEmpty:
                    break
            break
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                async with client.stream(
                    "POST",
                    f"https://api.elevenlabs.io/v1/text-to-speech/{vid}/stream",
                    headers={
                        "xi-api-key": ELEVENLABS_API_KEY,
                        "Content-Type": "application/json",
                    },
                    params={"output_format": output_format} if output_format else None,
                    json={"text": sentence, **tts_body},
                ) as resp:
                    resp.raise_for_status()
                    audio_data = b""
                    async for chunk in resp.aiter_bytes(4096):
                        if interrupt.is_set():
                            break
                        if chunk:
                            audio_data += chunk
                    if audio_data and not interrupt.is_set():
                        await transport.send_bytes(audio_data)
                        audio_sent = True
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"TTS error for '{sentence[:30]}': {e}")
    return audio_sent


# ---------------------------------------------------------------- phone leg
# VoiceLink's media-stream protocol mirrors Twilio's: JSON text frames with
# base64 A-law payloads. Docs: app.voicelink.co.in/documentation/

class PhoneTransport:
    """Looks like a WebSocket to run_turn/tts_consumer, speaks VoiceLink."""

    SEGMENT = 8000   # 1s of 8kHz A-law; VoiceLink wants segments under 2s

    def __init__(self, ws: WebSocket):
        self.ws         = ws
        self.stream_sid = None
        self.call_sid   = None

    async def send_json(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "interrupted":
            await self.clear()
        elif kind == "tool_call":
            print(f"[phone] tool {msg.get('name')} -> {str(msg.get('result'))[:80]}")
        elif kind in ("user_transcript", "agent_transcript", "error"):
            print(f"[phone] {kind}: {str(msg.get('text'))[:100]}")

    async def send_bytes(self, audio: bytes) -> None:
        # shortcut: unpaced, as Twilio accepts — the carrier buffers and plays at
        # 8kHz. If VoiceLink needs paced frames, meter this to 1s of audio/second.
        for i in range(0, len(audio), self.SEGMENT):
            await self.ws.send_json({
                "event": "media",
                "media": {"payload": base64.b64encode(audio[i:i + self.SEGMENT]).decode()},
            })

    async def clear(self) -> None:
        """Drop audio already queued at the carrier — the half of barge-in that
        setting interrupt_event cannot reach."""
        if self.stream_sid:
            await self.ws.send_json({"event": "clear", "stream_sid": self.stream_sid})

    async def transfer(self, target: str) -> None:
        await self.ws.send_json({"event": "transfer", "target": target})


@app.post("/voicelink/inbound")
async def voicelink_inbound(req: Request):
    """VoiceLink asks us, per inbound call, where to stream it. Both keys required."""
    data = await req.json()
    print(f"[phone] inbound call={data.get('callId')} sid={data.get('sid')} "
          f"from={data.get('customerNumber')} did={data.get('callerId')}")
    if not PUBLIC_BASE_URL:
        print("[phone] PUBLIC_BASE_URL unset — VoiceLink cannot reach us")
    ws_base = PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    call_id = data.get("callId") or data.get("sid") or ""
    return {
        "ws_url":      f"{ws_base}/ws/phone?call_id={call_id}",
        "webhook_url": f"{PUBLIC_BASE_URL}/voicelink/webhook",
    }


@app.post("/voicelink/webhook")
async def voicelink_webhook(req: Request):
    """Call lifecycle. VoiceLink documents two payload shapes; log both spellings."""
    data = await req.json()
    print(f"[phone] webhook {data.get('event') or 'status'} call={data.get('callId')} "
          f"status={data.get('callStatus')} "
          f"dur={data.get('duration') or data.get('durationSeconds')} "
          f"by={data.get('disconnectedBy')} rec={data.get('recordingUrl')}")
    return {"ok": True}


@app.websocket("/ws/phone")
async def phone_endpoint(ws: WebSocket):
    await ws.accept()
    print(f"[phone] stream open call_id={ws.query_params.get('call_id', '')}")

    transport = PhoneTransport(ws)
    session   = Session(transport)
    session.tts_format = "alaw_8000"
    await session.apply_settings({"tools": "telecom",
                                  "system_prompt": telecom_tools.SYSTEM_PROMPT})

    async def on_vad(event: str) -> None:
        # Caller started talking. If the agent is mid-answer that is a barge-in:
        # stop generating AND drop what the carrier already buffered.
        if event == "vad.speech_start" and session.active_turn and not session.active_turn.done():
            print("[phone] barge-in")
            await session.on_interrupt()
            await transport.clear()

    async def drive_turns() -> None:
        """Sarvam owns the turn boundaries here, so transcripts drive the loop."""
        while True:
            text, lang = await session.transcript_queue.get()
            if text.strip():
                await session.start_turn(text, lang)

    stt_ok = await session.open_stt(on_vad=on_vad, endpointing="vad",
                                   encoding="alaw", sample_rate=8000)
    if not stt_ok:
        print("[phone] no STT — caller will hear the greeting and nothing after")
    driver = asyncio.create_task(drive_turns()) if stt_ok else None

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[phone] dropped non-JSON frame: {raw[:80]}")
                continue

            event = msg.get("event")
            if event == "media":
                await session.feed_audio_b64(msg.get("media", {}).get("payload", ""))
            elif event == "start":
                start = msg.get("start", {})
                # casing differs across their events: stream_sid/call_sid vs callSid
                transport.stream_sid = start.get("stream_sid") or msg.get("stream_sid")
                transport.call_sid   = start.get("call_sid")
                print(f"[phone] start call={transport.call_sid} from={start.get('from')} "
                      f"to={start.get('to')} fmt={start.get('media_format')}")
                await session.greet()
            elif event == "stop":
                print(f"[phone] stop call={msg.get('stop', {}).get('callSid')}")
                break
            elif event == "connected":
                print("[phone] connected")
            elif event == "mark":
                print(f"[phone] mark {msg.get('mark', {}).get('name')}")
    except WebSocketDisconnect:
        print("[phone] carrier disconnected")
    except Exception as e:
        print(f"[phone] error: {e}")
    finally:
        if driver:
            driver.cancel()
        await session.close()
