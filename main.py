import asyncio
import os
import json
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

SARVAM_API_KEY      = os.getenv("SARVAM_API_KEY")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

SYSTEM_PROMPT = (
    "You are a helpful and friendly voice assistant. "
    "Respond in the same language the user speaks. "
    "For simple conversational questions, keep it brief (1-2 sentences). "
    "For factual or informational questions — like travel costs, routes, prices, comparisons — "
    "give a complete and specific answer: include actual numbers, options (bus/train/flight), "
    "and relevant details. Do not hedge with vague phrases like 'it depends' unless you truly "
    "cannot give any estimate. Speak naturally as if talking to a friend."
)

SENTENCE_ENDERS = frozenset('.!?।\n')
MIN_SENTENCE_LEN = 15


def flush_sentences(buffer: str) -> tuple[list[str], str]:
    """Return (complete_sentences, remaining_buffer)."""
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


@app.get("/")
async def get():
    return HTMLResponse(open("index.html").read())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected")

    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    audio_mime   = "audio/webm"
    voice_id     = ELEVENLABS_VOICE_ID
    system_prompt = SYSTEM_PROMPT

    try:
        while True:
            msg = await ws.receive()

            if "text" in msg:
                data = json.loads(msg["text"])

                if data.get("type") == "audio_mime":
                    audio_mime = data["mime"]
                    print(f"Mime set: {audio_mime}")

                elif data.get("type") == "settings":
                    voice_id      = data.get("voice_id", voice_id)
                    system_prompt = data.get("system_prompt", system_prompt)
                    conversation_history = [{"role": "system", "content": system_prompt}]
                    print(f"Settings updated — voice: {voice_id}")
                    print(f"System prompt: {system_prompt[:80]}...")
                    await ws.send_json({"type": "settings_ack"})

                continue

            if "bytes" not in msg or not msg["bytes"]:
                continue

            audio_bytes = msg["bytes"]
            print(f"Audio received: {len(audio_bytes)} bytes ({audio_mime})")

            # ── Step 1: STT ──────────────────────────────────────
            user_text = await speech_to_text(audio_bytes, audio_mime)
            if not user_text or not user_text.strip():
                await ws.send_json({"type": "error", "text": "Samajh nahi aaya, dobara bolo."})
                await ws.send_json({"type": "ready"})
                continue

            print(f"[USER] {user_text}")
            await ws.send_json({"type": "user_transcript", "text": user_text})

            # ── Steps 2+3: LLM + TTS concurrent ──────────────────
            conversation_history.append({"role": "user", "content": user_text})

            sentence_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
            results = await asyncio.gather(
                stream_llm_producer(conversation_history, sentence_queue),
                tts_consumer(sentence_queue, ws, voice_id),
                return_exceptions=True,
            )
            full_text   = results[0] if isinstance(results[0], str) else ""
            audio_sent  = results[1] if isinstance(results[1], bool) else False

            if not full_text:
                await ws.send_json({"type": "tts_end"})
                await ws.send_json({"type": "error", "text": "LLM ne respond nahi kiya."})
                continue

            conversation_history.append({"role": "assistant", "content": full_text})
            print(f"[AGENT] {full_text}")
            await ws.send_json({"type": "agent_transcript", "text": full_text})

            if audio_sent:
                await ws.send_json({"type": "tts_end"})
            else:
                # ElevenLabs failed — fallback to browser TTS
                await ws.send_json({"type": "tts_fallback", "text": full_text})

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


async def speech_to_text(audio_bytes: bytes, mime: str = "audio/webm") -> str:
    base_mime = mime.split(";")[0].strip()
    ext_map = {
        "audio/webm": "webm",
        "audio/mp4":  "mp4",
        "audio/ogg":  "ogg",
        "audio/wav":  "wav",
    }
    ext = ext_map.get(base_mime, "webm")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": SARVAM_API_KEY},
                files={"file": (f"audio.{ext}", audio_bytes, mime)},
                data={"language_code": "hi-IN", "model": "saaras:v3"},
            )
            if not response.is_success:
                print(f"STT {response.status_code}: {response.text}")
                return ""
            result = response.json()
            print(f"Sarvam: {result}")
            return result.get("transcript", "")
    except Exception as e:
        print(f"STT error: {e}")
        return ""


async def stream_llm_producer(history: list, queue: asyncio.Queue) -> str:
    """Stream Groq tokens, split into sentences, push to queue. Returns full text."""
    full_text = ""
    buffer = ""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": history,
                    "max_tokens": 600,
                    "stream": True,
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                        continue
                    try:
                        delta = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if not delta:
                        continue
                    full_text += delta
                    buffer += delta
                    sentences, buffer = flush_sentences(buffer)
                    for s in sentences:
                        await queue.put(s)
        if buffer.strip():
            await queue.put(buffer.strip())
    except Exception as e:
        print(f"LLM stream error: {e}")
    finally:
        await queue.put(None)
    return full_text


async def tts_consumer(queue: asyncio.Queue, ws: WebSocket, vid: str) -> bool:
    """Consume sentences from queue, collect full audio per sentence, send as one frame.
    Returns True if at least one audio chunk was sent successfully."""
    vid = vid or ELEVENLABS_VOICE_ID
    audio_sent = False
    while True:
        sentence = await queue.get()
        if sentence is None:
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
                        if chunk:
                            audio_data += chunk
                    if audio_data:
                        await ws.send_bytes(audio_data)
                        audio_sent = True
        except Exception as e:
            print(f"TTS error for '{sentence[:30]}': {e}")
    return audio_sent
