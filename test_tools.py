"""Self-check for banking tools + streamed tool-call reassembly. Run: python3 test_tools.py"""
import json

from banking_tools import new_account, get_balance, recent_transactions, block_card, execute_tool
from main import accumulate_tool_calls

# tool functions
acc = new_account()
assert "₹42,300" in get_balance(acc)
assert recent_transactions(acc, 3).count(";") == 2          # exactly 3 items
assert recent_transactions(acc, 99).count(";") == 4         # clamped to 5
assert recent_transactions(acc, -1).count(";") == 0         # clamped to 1
assert not acc["card_blocked"]
assert "blocked" in block_card(acc).lower() and acc["card_blocked"]
assert "already" in block_card(acc).lower()                 # idempotent
assert "blocked" in get_balance(acc)                        # mutation visible

# per-connection isolation
acc2 = new_account()
assert not acc2["card_blocked"]

# dispatch error paths — never raise
assert execute_tool("nope", "{}", acc).startswith("error")
assert execute_tool("recent_transactions", "{bad json", acc).startswith("error")
assert execute_tool("recent_transactions", '{"n": "x"}', acc).startswith("error")
assert execute_tool("get_balance", "", acc).startswith("Balance")
assert "₹" in execute_tool("recent_transactions", '{"n": 2}', acc)

# reassembly (critical): name whole in first chunk, args fragmented across chunks
buf = {}
accumulate_tool_calls(buf, [{"index": 0, "id": "call_1", "function": {"name": "recent_transactions"}}])
accumulate_tool_calls(buf, [{"index": 0, "function": {"arguments": '{"n"'}}])
accumulate_tool_calls(buf, [{"index": 0, "function": {"arguments": ": 3}"}}])
assert buf[0]["id"] == "call_1"
assert buf[0]["name"] == "recent_transactions"
assert json.loads(buf[0]["args"]) == {"n": 3}

# two interleaved calls keep separate slots
buf = {}
accumulate_tool_calls(buf, [{"index": 0, "id": "a", "function": {"name": "get_balance", "arguments": ""}},
                            {"index": 1, "id": "b", "function": {"name": "block_card"}}])
accumulate_tool_calls(buf, [{"index": 1, "function": {"arguments": "{}"}}])
assert buf[0]["name"] == "get_balance" and buf[1]["name"] == "block_card" and buf[1]["args"] == "{}"


# --- telecom tools ---
import telecom_tools as tt

t = tt.new_account()
assert "Smart 299" in tt.get_account(t)
assert tt.get_recent_charges(t, 99).count(";") == 4                 # clamped to 5
assert "Caller Tune" in tt.get_recent_charges(t)                    # the "paise kat gaye" culprit
assert "outage" in tt.check_outage(t, "Sector 15 market").lower()   # substring match
assert "No known outage" in tt.check_outage(t, "Rajouri Garden")
assert "active" in tt.check_number_status(t, "98765 43210")         # own number, formatted
assert "recoverable until" in tt.check_number_status(t, "+91 9812304567")  # deactivated SIM
assert "prepaid" in tt.check_number_status(t, "9812304567")         # plan type preserved
assert "can't be shared" in tt.check_number_status(t, "9000000000") # privacy: unknown number

r = tt.register_complaint(t, "reactivation", "recover number ending 4567")
assert "TKT-4821" in r
assert "registered" in tt.get_ticket_status(t, "4821")              # bare ID normalized
assert "in progress" in tt.get_ticket_status(t, "tkt-4796")         # seeded ticket, case-insensitive
assert "No ticket" in tt.get_ticket_status(t, "TKT-9999")
assert "TKT-4822" in tt.register_complaint(t, "network", "slow data")  # counter increments
assert "executive" in tt.transfer_to_human(t, "caller asked for human")

# per-connection isolation + dispatch error path
t2 = tt.new_account()
assert "TKT-4821" not in t2["tickets"]
assert tt.execute_tool("nope", "{}", t2).startswith("error")
assert tt.execute_tool("check_outage", "{bad", t2).startswith("error")
assert "Smart 299" in tt.execute_tool("get_account", "", t2)

# normalize_args: Llama's `null`/empty args for no-arg tools must become {}
from main import normalize_args
assert normalize_args("null") == "{}"
assert normalize_args("") == "{}"
assert normalize_args(' {"n": 3}') == ' {"n": 3}'

# strip_fn_tags: paired tags, orphan close (arrives token-split in stream), space preserved
from main import strip_fn_tags
assert strip_fn_tags("<function=get_balance>{}</function>") == ""
assert strip_fn_tags("theek hai? </function>") == "theek hai? "
assert strip_fn_tags("open from 8am") == "open from 8am"

# salvage_failed_tool_call: Groq 400 tool_use_failed → recovered call, else None
from main import salvage_failed_tool_call
err = json.dumps({"error": {"code": "tool_use_failed",
    "failed_generation": '<function=recent_transactions[]{"n": "3"}</function>'}})
c = salvage_failed_tool_call(err)
assert c["name"] == "recent_transactions" and json.loads(c["args"]) == {"n": "3"}
c = salvage_failed_tool_call(json.dumps({"error": {"failed_generation": "<function=get_balance></function>"}}))
assert c["name"] == "get_balance" and c["args"] == "{}"
assert salvage_failed_tool_call('{"error":{"message":"rate limit"}}') is None
assert salvage_failed_tool_call("not json at all") is None

# --- Session: the refactor's fragile parts (stubbed LLM/TTS, no network) ---
import asyncio
import main


class _FakeTransport:
    def __init__(self):
        self.sent = []
    async def send_json(self, m):
        self.sent.append(m)
    async def send_bytes(self, b):
        pass

    @property
    def types(self):
        return [m["type"] for m in self.sent]


def _stub(interrupt_mid_turn: bool, text: str = "haan ji"):
    """Replace the LLM/TTS pair with fakes that mimic a tool-using turn."""
    async def producer(history, queue, interrupt, *a, **k):
        history.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_balance", "arguments": "{}"}}]})
        history.append({"role": "tool", "tool_call_id": "c1", "content": "Balance ₹42,300"})
        if interrupt_mid_turn:
            interrupt.set()
        await queue.put(None)
        return text
    async def consumer(queue, *a, **k):
        while await queue.get() is not None:
            pass
        return True
    main.stream_llm_producer, main.tts_consumer = producer, consumer


# interrupt mid-turn: the WHOLE turn is rolled back, not just the last message.
# Leaving a dangling tool message here 400s the next Groq request.
_stub(interrupt_mid_turn=True)
s = main.Session(_FakeTransport())
base = len(s.history)
asyncio.run(s.run_turn("mera balance batao"))
assert len(s.history) == base, f"turn not rolled back: {s.history}"
assert "interrupted" in s.transport.types
assert "agent_transcript" not in s.transport.types

# clean turn: user + tool messages + assistant reply all survive
_stub(interrupt_mid_turn=False, text="Aapka balance ₹42,300 hai.")
s = main.Session(_FakeTransport())
base = len(s.history)
asyncio.run(s.run_turn("mera balance batao"))
assert len(s.history) == base + 4                       # user, tool_calls, tool, assistant
assert s.history[-1] == {"role": "assistant", "content": "Aapka balance ₹42,300 hai."}
assert s.transport.types == ["user_transcript", "agent_transcript", "tts_end"]

# empty transcript never starts a turn
s = main.Session(_FakeTransport())
asyncio.run(s.run_turn("   "))
assert len(s.history) == 1 and s.transport.types == ["error", "ready"]

# apply_settings selects the toolset atomically and resets history
s = main.Session(_FakeTransport())
asyncio.run(s.apply_settings({"tools": "telecom", "voice_id": "v1", "system_prompt": "P"}))
assert s.exec_tool is tt.execute_tool and s.account["plan"]
assert s.history == [{"role": "system", "content": "P"}] and s.voice_id == "v1"
asyncio.run(s.apply_settings({"tools": "nonexistent"}))
assert (s.active_tools, s.account, s.exec_tool) == (None, None, None)   # degrades to no-tools


# --- sarvam_reader: empty finals must reach the queue (else 10s of dead air) ---
class _FakeSarvamWS:
    def __init__(self, msgs):
        self._msgs = [json.dumps(m) for m in msgs]
    def __aiter__(self):
        async def gen():
            for m in self._msgs:
                yield m
        return gen()


def _drain(msgs):
    q = asyncio.Queue()
    asyncio.run(main.sarvam_reader(_FakeSarvamWS(msgs), q))
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out

# empty text still queues, so on_speech_end fails fast instead of timing out
assert _drain([{"event": "transcript.final", "utterance_idx": 0, "text": "", "language": "en-IN"}]) \
    == [("", "en-IN")]
# normal transcript carries the hyphenated language through for ElevenLabs
assert _drain([{"event": "transcript.final", "text": "mera balance batao", "language": "hi-IN"}]) \
    == [("mera balance batao", "hi-IN")]
# non-transcript events are not queued
assert _drain([{"event": "session.begin", "request_id": "x", "config": {}},
               {"event": "error", "code": "chunk_too_large", "is_fatal": False, "message": "m"},
               {"event": "pong"}]) == []


print("all checks passed")
