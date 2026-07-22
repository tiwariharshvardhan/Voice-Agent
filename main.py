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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from banking_tools import BANKING_TOOLS, new_account, execute_tool

load_dotenv()

app = FastAPI()

SARVAM_API_KEY      = os.getenv("SARVAM_API_KEY")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "bajNon13EdhNMndG3z05")

SARVAM_STT_WS = "wss://api.sarvam.ai/speech-to-text/ws"

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


async def connect_sarvam_ws(language_code: str):
    """Open a Sarvam STT WebSocket with config in URL params."""
    params = {
        "model": "saaras:v3",
        "mode": "transcribe",
        "sample_rate": "16000",
        "input_audio_codec": "pcm_s16le",
        "flush_signal": "true",
    }
    if language_code and language_code != "unknown":
        params["language-code"] = language_code
    url = f"{SARVAM_STT_WS}?{urlencode(params)}"
    if DEBUG: print(f"Connecting to Sarvam: {url}")
    return await ws_connect(url, additional_headers={"api-subscription-key": SARVAM_API_KEY}, ssl=SSL_CTX)


async def sarvam_reader(sarvam_ws, transcript_queue: asyncio.Queue):
    """Background task: drain Sarvam responses and push transcripts to queue."""
    try:
        async for msg in sarvam_ws:
            data = json.loads(msg)
            if DEBUG: print(f"Sarvam msg: {data}")
            if data.get("type") == "data":
                transcript = data.get("data", {}).get("transcript", "")
                if transcript:
                    await transcript_queue.put(transcript)
    except Exception as e:
        print(f"Sarvam reader closed: {e}")


@app.get("/")
async def get():
    return HTMLResponse(open("index.html").read())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected")

    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    voice_id        = ELEVENLABS_VOICE_ID
    system_prompt   = SYSTEM_PROMPT
    language_code   = "unknown"
    active_turn     = None
    active_tools    = None
    account         = new_account()   # per-connection state — sessions never share
    interrupt_event = asyncio.Event()
    transcript_queue = asyncio.Queue()

    # Sarvam WS opened per-utterance on speech_start (avoids idle timeout)
    sarvam_ws   = None
    reader_task = None

    async def run_turn(transcript: str) -> None:
        if not transcript.strip():
            await ws.send_json({"type": "error", "text": "Samajh nahi aaya, dobara bolo."})
            await ws.send_json({"type": "ready"})
            return

        print(f"[USER] {transcript}")
        await ws.send_json({"type": "user_transcript", "text": transcript})
        turn_start = len(conversation_history)
        conversation_history.append({"role": "user", "content": transcript})

        sentence_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
        results = await asyncio.gather(
            stream_llm_producer(conversation_history, sentence_queue, interrupt_event,
                                active_tools, account, ws),
            tts_consumer(sentence_queue, ws, voice_id, interrupt_event),
            return_exceptions=True,
        )
        full_text  = results[0] if isinstance(results[0], str) else ""
        audio_sent = results[1] if isinstance(results[1], bool) else False

        if interrupt_event.is_set():
            print(f"[INTERRUPTED] {full_text[:60]}")
            # Drop the whole turn — including any tool-call/tool messages the
            # hop-loop appended — so history never ends on a dangling tool msg.
            del conversation_history[turn_start:]
            await ws.send_json({"type": "interrupted"})
            return

        if full_text:
            conversation_history.append({"role": "assistant", "content": full_text})
        print(f"[AGENT] {full_text}")
        await ws.send_json({"type": "agent_transcript", "text": full_text})

        if audio_sent:
            await ws.send_json({"type": "tts_end"})
        else:
            await ws.send_json({"type": "tts_fallback", "text": full_text})

    def on_turn_done(task) -> None:
        try:
            task.result()
        except (asyncio.CancelledError, Exception):
            pass

    try:
        while True:
            msg = await ws.receive()

            if "text" in msg:
                data = json.loads(msg["text"])

                if data.get("type") == "settings":
                    language_code = data.get("language_code", language_code)
                    voice_id      = data.get("voice_id", voice_id)
                    system_prompt = data.get("system_prompt", system_prompt)
                    active_tools  = BANKING_TOOLS if data.get("tools") == "banking" else None
                    conversation_history = [{"role": "system", "content": system_prompt}]
                    print(f"Settings updated — voice: {voice_id}, lang: {language_code}, tools: {'banking' if active_tools else 'none'}")
                    await ws.send_json({"type": "settings_ack"})

                elif data.get("type") == "speech_start":
                    print("STT stream starting")
                    while not transcript_queue.empty():
                        transcript_queue.get_nowait()
                    # Open fresh Sarvam WS per utterance — avoids idle timeout
                    sarvam_ws   = await connect_sarvam_ws(language_code)
                    reader_task = asyncio.create_task(sarvam_reader(sarvam_ws, transcript_queue))

                elif data.get("type") == "speech_end":
                    print("STT stream ending — flushing Sarvam")
                    await sarvam_ws.send(json.dumps({"type": "flush"}))

                    try:
                        transcript = await asyncio.wait_for(transcript_queue.get(), timeout=10.0)
                    except asyncio.TimeoutError:
                        transcript = ""

                    reader_task.cancel()
                    await sarvam_ws.close()
                    sarvam_ws = None

                    if not transcript:
                        await ws.send_json({"type": "error", "text": "Samajh nahi aaya, dobara bolo."})
                        await ws.send_json({"type": "ready"})
                        continue

                    if active_turn and not active_turn.done():
                        interrupt_event.set()
                        active_turn.cancel()
                        try: await active_turn
                        except (asyncio.CancelledError, Exception): pass

                    interrupt_event.clear()
                    active_turn = asyncio.create_task(run_turn(transcript))
                    active_turn.add_done_callback(on_turn_done)

                elif data.get("type") == "speech_cancel":
                    print("STT cancelled (too short)")
                    if reader_task:
                        reader_task.cancel()
                    if sarvam_ws:
                        await sarvam_ws.close()
                        sarvam_ws = None
                    while not transcript_queue.empty():
                        transcript_queue.get_nowait()
                    await ws.send_json({"type": "ready"})

                elif data.get("type") == "interrupt":
                    print("Interrupt received")
                    if active_turn and not active_turn.done():
                        interrupt_event.set()

                continue

            if "bytes" not in msg or not msg["bytes"]:
                continue

            # Binary PCM chunk — forward to Sarvam in real-time
            audio_b64 = base64.b64encode(msg["bytes"]).decode()
            await sarvam_ws.send(json.dumps({
                "audio": {"data": audio_b64, "sample_rate": 16000, "encoding": "audio/wav"},
            }))

    except WebSocketDisconnect:
        print("Client disconnected")
        if active_turn and not active_turn.done():
            active_turn.cancel()
    except RuntimeError as e:
        if "disconnect" in str(e).lower():
            print("Client disconnected mid-session")
        else:
            print(f"Runtime error: {e}")
        if active_turn and not active_turn.done():
            active_turn.cancel()
    except Exception as e:
        print(f"Unhandled error: {e}")
        if active_turn and not active_turn.done():
            active_turn.cancel()
        try:
            await ws.send_json({"type": "error", "text": str(e)})
        except Exception:
            pass
    finally:
        if reader_task:
            reader_task.cancel()
        if sarvam_ws:
            try:
                await sarvam_ws.close()
            except Exception:
                pass


MAX_HOPS = 4  # ponytail: fixed hop cap; raise only if a real multi-tool flow needs it


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
        "model": "llama-3.3-70b-versatile",
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
                content = re.sub(r"<function=\w+>.*?</function>", "", content, flags=re.DOTALL)
                if not content.strip():
                    continue
                text += content
                buffer += content
                sentences, buffer = flush_sentences(buffer)
                for s in sentences:
                    if interrupt.is_set():
                        break
                    await queue.put(s)
    if buffer.strip() and not interrupt.is_set():
        await queue.put(buffer.strip())
    return text, [calls[i] for i in sorted(calls)]


async def stream_llm_producer(history: list, queue: asyncio.Queue, interrupt: asyncio.Event,
                              tools=None, account=None, ws=None) -> str:
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
                result = execute_tool(tc["name"], tc["args"], account)
                print(f"[TOOL] {tc['name']}({tc['args']}) -> {result}")
                if ws is not None:
                    await ws.send_json({"type": "tool_call", "name": tc["name"], "result": result})
                history.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"LLM stream error: {e}")
    finally:
        await queue.put(None)
    return full_text


async def tts_consumer(queue: asyncio.Queue, ws: WebSocket, vid: str, interrupt: asyncio.Event) -> bool:
    vid = vid or ELEVENLABS_VOICE_ID
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
                    json={
                        "text": sentence,
                        "model_id": "eleven_flash_v2_5",
                        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                    },
                ) as resp:
                    resp.raise_for_status()
                    audio_data = b""
                    async for chunk in resp.aiter_bytes(4096):
                        if interrupt.is_set():
                            break
                        if chunk:
                            audio_data += chunk
                    if audio_data and not interrupt.is_set():
                        await ws.send_bytes(audio_data)
                        audio_sent = True
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"TTS error for '{sentence[:30]}': {e}")
    return audio_sent
