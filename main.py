import asyncio
import io
import os
import json
import ssl
import wave
import certifi
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

SSL_CTX = ssl.create_default_context(cafile=certifi.where())

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
    voice_id        = ELEVENLABS_VOICE_ID
    system_prompt   = SYSTEM_PROMPT
    language_code   = "unknown"
    active_turn     = None
    interrupt_event = asyncio.Event()

    # STT buffering state
    audio_buffer  = []
    is_stt_active = False

    async def run_turn(transcript: str) -> None:
        if not transcript.strip():
            await ws.send_json({"type": "error", "text": "Samajh nahi aaya, dobara bolo."})
            await ws.send_json({"type": "ready"})
            return

        print(f"[USER] {transcript}")
        await ws.send_json({"type": "user_transcript", "text": transcript})
        conversation_history.append({"role": "user", "content": transcript})

        sentence_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
        results = await asyncio.gather(
            stream_llm_producer(conversation_history, sentence_queue, interrupt_event),
            tts_consumer(sentence_queue, ws, voice_id, interrupt_event),
            return_exceptions=True,
        )
        full_text  = results[0] if isinstance(results[0], str) else ""
        audio_sent = results[1] if isinstance(results[1], bool) else False

        if interrupt_event.is_set():
            print(f"[INTERRUPTED] {full_text[:60]}")
            if conversation_history and conversation_history[-1]["role"] == "user":
                conversation_history.pop()
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
                    voice_id      = data.get("voice_id", voice_id)
                    system_prompt = data.get("system_prompt", system_prompt)
                    language_code = data.get("language_code", language_code)
                    conversation_history = [{"role": "system", "content": system_prompt}]
                    print(f"Settings updated — voice: {voice_id}, lang: {language_code}")
                    print(f"System prompt: {system_prompt[:80]}...")
                    await ws.send_json({"type": "settings_ack"})

                elif data.get("type") == "speech_start":
                    print("STT stream starting — buffering PCM")
                    is_stt_active = True
                    audio_buffer.clear()

                elif data.get("type") == "speech_end":
                    print(f"STT stream ending — {len(audio_buffer)} chunks buffered")
                    is_stt_active = False
                    if not audio_buffer:
                        await ws.send_json({"type": "error", "text": "Samajh nahi aaya, dobara bolo."})
                        await ws.send_json({"type": "ready"})
                        continue

                    pcm_data = b"".join(audio_buffer)
                    audio_buffer.clear()
                    transcript = await speech_to_text(pcm_data, language_code)

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
                    is_stt_active = False
                    audio_buffer.clear()
                    await ws.send_json({"type": "ready"})

                elif data.get("type") == "interrupt":
                    print("Interrupt received")
                    if active_turn and not active_turn.done():
                        interrupt_event.set()

                continue

            if "bytes" not in msg or not msg["bytes"]:
                continue

            # Binary = PCM chunk from AudioWorklet — buffer it
            if is_stt_active:
                audio_buffer.append(msg["bytes"])

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


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


async def speech_to_text(pcm_data: bytes, language_code: str = "unknown") -> str:
    wav_data = pcm_to_wav(pcm_data)
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            data = {"model": "saaras:v3"}
            if language_code and language_code != "unknown":
                data["language_code"] = language_code
            response = await client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": SARVAM_API_KEY},
                files={"file": ("audio.wav", wav_data, "audio/wav")},
                data=data,
            )
            if not response.is_success:
                print(f"STT {response.status_code}: {response.text}")
                return ""
            result = response.json()
            print(f"Sarvam STT: {result}")
            return result.get("transcript", "")
    except Exception as e:
        print(f"STT error: {e}")
        return ""


async def stream_llm_producer(history: list, queue: asyncio.Queue, interrupt: asyncio.Event) -> str:
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
                    if interrupt.is_set():
                        break
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
                        if interrupt.is_set():
                            break
                        await queue.put(s)
        if buffer.strip() and not interrupt.is_set():
            await queue.put(buffer.strip())
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
