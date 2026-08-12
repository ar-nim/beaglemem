"""Shared synthetic data."""


def make_chat_jsonl_lines():
    return [
        '{"role": "user", "content": "I got let go from beta division. The severance letter came today."}',
        '{"role": "assistant", "content": [{"type": "text", "text": "Noted. The restructuring is confirmed."}]}',
        '{"role": "tool", "content": "tool output should be ignored entirely"}',
        '{"role": "user", "content": ""}',
        '{"role": "assistant", "content": [{"type": "tool_use", "name": "terminal"}]}',
        'not valid json at all',
    ]


SYNTHETIC_SENTENCES = [
    ["termination", "contract", "signed", "hr"],
    ["contract", "signed", "termination", "letter"],
    ["retrenched", "contract", "signed", "hr"],
    ["hr", "retrenched", "letter", "signed"],
    ["weather", "sunny", "weekend", "forecast"],
    ["sunny", "forecast", "weather", "weekend"],
]
