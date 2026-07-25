"""Telecom customer-care toolset. Mock data, per-connection state.
Design: docs/telecom-agent-design.md"""
import copy
import json

ACCOUNT_TEMPLATE = {
    "name": "Rahul Sharma",
    "number": "9876543210",
    "plan": {"name": "Smart 299", "price": 299, "data_per_day": "1.5GB",
             "validity_end": "2026-08-14", "type": "prepaid"},
    "balance": 12.50,
    "data_left_today": "0.4GB",
    "charges": [
        {"date": "2026-07-24", "desc": "Recharge Smart 299", "amount": 299, "type": "credit"},
        {"date": "2026-07-22", "desc": "Caller Tune subscription (auto-renewal)", "amount": 35, "type": "debit"},
        {"date": "2026-07-20", "desc": "Data booster pack 1GB", "amount": 19, "type": "debit"},
        {"date": "2026-07-15", "desc": "International SMS x3", "amount": 12, "type": "debit"},
        {"date": "2026-07-08", "desc": "Recharge Smart 299", "amount": 299, "type": "credit"},
    ],
    # Second SIM on the same registered ID — deactivated, still in recovery window.
    "other_numbers": {
        "9812304567": {"status": "deactivated", "deactivated_on": "2026-06-30",
                       "recoverable_until": "2026-09-28", "plan_type": "prepaid"},
    },
    "tickets": {
        "TKT-4796": {"category": "network", "description": "Call drops in Sector 12",
                     "status": "in progress", "eta": "2026-07-26, 6 PM"},
    },
    "next_ticket": 4821,
}

# ponytail: static outage table; wire a real status API if this ever leaves demo
OUTAGES = {"sector 15": "Fiber cut — restoration expected today by 6 PM"}


def new_account():
    """Fresh per-connection copy — concurrent sessions never share state."""
    return copy.deepcopy(ACCOUNT_TEMPLATE)


def get_account(account):
    p = account["plan"]
    return (f"Plan: {p['name']} ({p['type']}), ₹{p['price']}, {p['data_per_day']}/day, "
            f"valid till {p['validity_end']}. Talktime balance: ₹{account['balance']}. "
            f"Data left today: {account['data_left_today']}.")


def get_recent_charges(account, n=5):
    n = max(1, min(int(n), 5))  # never trust model-supplied sizes
    return "; ".join(
        f"{c['date']}: {c['desc']} ₹{c['amount']} ({c['type']})"
        for c in account["charges"][:n]
    )


def check_outage(account, area):
    area_l = str(area).lower().strip()
    for key, info in OUTAGES.items():
        if key in area_l or area_l in key:
            return f"Known outage in {area}: {info}."
    return f"No known outage in {area}. The issue is likely device- or SIM-specific."


def check_number_status(account, number):
    digits = "".join(ch for ch in str(number) if ch.isdigit())[-10:]
    if digits == account["number"]:
        return "That is this number — it is active."
    info = account["other_numbers"].get(digits)
    if info is None:
        # trust boundary: never reveal status of numbers not on the caller's ID
        return ("This number is not linked to your registered ID, so its details "
                "can't be shared on this call. The registered owner can call us, "
                "or visit a store with their ID proof.")
    return (f"Number ending {digits[-4:]}: {info['status']} since {info['deactivated_on']}, "
            f"recoverable until {info['recoverable_until']} with the same "
            f"{info['plan_type']} plan type. Recovery: one store visit with ID proof, "
            f"or register a reactivation request on this call.")


def register_complaint(account, category, description):
    tid = f"TKT-{account['next_ticket']}"
    account["next_ticket"] += 1
    account["tickets"][tid] = {"category": category, "description": description,
                               "status": "registered", "eta": "within 48 hours"}
    return f"Complaint registered. Ticket ID {tid}, resolution expected within 48 hours."


def get_ticket_status(account, ticket_id):
    tid = str(ticket_id).upper().strip()
    if not tid.startswith("TKT-"):
        tid = f"TKT-{tid.lstrip('TKT').strip('-')}"
    t = account["tickets"].get(tid)
    if t is None:
        return f"No ticket found with ID {tid}. Please confirm the number."
    return f"{tid} ({t['category']}): {t['status']}, expected resolution {t['eta']}."


def transfer_to_human(account, reason=""):
    return ("Transfer initiated — connecting the customer to a customer care "
            "executive now. Tell them the expected wait is under 2 minutes, "
            "then stop; the executive takes over.")


TOOL_FUNCS = {
    "get_account": get_account,
    "get_recent_charges": get_recent_charges,
    "check_outage": check_outage,
    "check_number_status": check_number_status,
    "register_complaint": register_complaint,
    "get_ticket_status": get_ticket_status,
    "transfer_to_human": transfer_to_human,
}


def execute_tool(name, args_json, account):
    """Dispatch a tool call. Any failure (unknown name, garbled/wrong args)
    returns an error string so the model recovers next pass — never crashes the turn."""
    try:
        args = json.loads(args_json) if args_json and args_json.strip() else {}
        return str(TOOL_FUNCS[name](account, **args))
    except Exception:
        return "error: invalid tool or arguments"


def _fn(name, description, properties=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": required or []},
    }}


TELECOM_TOOLS = [
    _fn("get_account",
        "Get the caller's plan, price, validity, talktime balance and data left today. "
        "Use for any balance/plan/validity/data question, in any language, e.g. "
        "'mera plan kya hai', 'data kitna bacha hai', 'validity kab khatam hogi'."),
    _fn("get_recent_charges",
        "Get the caller's recent recharges and deductions. Use for 'money was deducted', "
        "'paise kat gaye', 'recharge nahi dikh raha', or any billing dispute.",
        {"n": {"type": "integer", "description": "How many entries, 1-5. Default 5."}}),
    _fn("check_outage",
        "Check for a known network outage in the caller's area. Ask for their area/locality "
        "first if not mentioned. Use for no-network, slow-data, call-drop complaints.",
        {"area": {"type": "string", "description": "Locality/area name as the caller said it."}},
        ["area"]),
    _fn("check_number_status",
        "Look up the status of a phone number — active, deactivated, recoverable. Use when "
        "the caller asks about a DIFFERENT number than the one they are calling from, e.g. "
        "an old SIM that went out of service, 'mera dusra number band ho gaya'.",
        {"number": {"type": "string", "description": "The phone number the caller is asking about."}},
        ["number"]),
    _fn("register_complaint",
        "Register a complaint/service-request ticket. ALWAYS read a one-line summary back "
        "to the caller and get a yes before calling this. Returns a ticket ID — read it "
        "out digit by digit.",
        {"category": {"type": "string", "description": "One of: network, billing, recharge, sim, reactivation, other."},
         "description": {"type": "string", "description": "One-line summary of the issue as confirmed with the caller."}},
        ["category", "description"]),
    _fn("get_ticket_status",
        "Get the status of an existing complaint ticket by its ID, e.g. 'TKT-4796' or '4796'.",
        {"ticket_id": {"type": "string", "description": "Ticket ID the caller provided."}},
        ["ticket_id"]),
    _fn("transfer_to_human",
        "Transfer the call to a human customer care executive. Use IMMEDIATELY when the "
        "caller asks for a human/real person, repeats the same issue twice, sounds "
        "frustrated, or the issue is outside your tools and knowledge. Never argue first.",
        {"reason": {"type": "string", "description": "Short reason for the transfer."}}),
]
