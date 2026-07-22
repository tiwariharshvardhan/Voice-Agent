import copy
import json

ACCOUNT_TEMPLATE = {
    "name": "Rahul Sharma",
    "balance": 42300,
    "card_blocked": False,
    "transactions": [
        {"date": "2026-07-18", "merchant": "Zomato", "amount": 480, "type": "debit"},
        {"date": "2026-07-17", "merchant": "Salary", "amount": 65000, "type": "credit"},
        {"date": "2026-07-16", "merchant": "Amazon", "amount": 1299, "type": "debit"},
        {"date": "2026-07-15", "merchant": "Uber", "amount": 240, "type": "debit"},
        {"date": "2026-07-14", "merchant": "DMart", "amount": 2150, "type": "debit"},
    ],
}


def new_account():
    """Fresh per-connection copy — concurrent sessions never share state."""
    return copy.deepcopy(ACCOUNT_TEMPLATE)


def get_balance(account):
    blocked = " (card is blocked)" if account["card_blocked"] else ""
    return f"Balance: ₹{account['balance']:,}{blocked}"


def recent_transactions(account, n=3):
    n = max(1, min(int(n), 5))  # never trust model-supplied sizes
    return "; ".join(
        f"{t['date']}: {t['merchant']} ₹{t['amount']:,} ({t['type']})"
        for t in account["transactions"][:n]
    )


def block_card(account):
    if account["card_blocked"]:
        return "Card is already blocked."
    account["card_blocked"] = True
    return "Card blocked successfully. A replacement card can be requested at any branch."


TOOL_FUNCS = {
    "get_balance": get_balance,
    "recent_transactions": recent_transactions,
    "block_card": block_card,
}


def execute_tool(name, args_json, account):
    """Dispatch a tool call. Any failure (unknown name, garbled/wrong args)
    returns an error string so the model recovers next pass — never crashes the turn."""
    try:
        args = json.loads(args_json) if args_json and args_json.strip() else {}
        return str(TOOL_FUNCS[name](account, **args))
    except Exception:
        return "error: invalid tool or arguments"


BANKING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_balance",
            "description": (
                "Get the customer's current account balance. The user may ask in any language, "
                "e.g. 'what's my balance', 'balance kitna hai', 'मेरा बैलेंस कितना है'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recent_transactions",
            "description": (
                "Get the customer's most recent transactions. The user may ask in any language, "
                "e.g. 'show my recent transactions', 'last teen transactions batao', "
                "'मेरे पिछले transactions दिखाओ'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "How many transactions to fetch, 1-5. Default 3."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "block_card",
            "description": (
                "Immediately block the customer's card. Use when the user reports a lost or stolen "
                "card or asks to block/freeze it, in any language, e.g. 'block my card', "
                "'card block karo', 'मेरा कार्ड खो गया है, ब्लॉक कर दो'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]
