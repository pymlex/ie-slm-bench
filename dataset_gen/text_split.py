from __future__ import annotations

import re

THINK_CLOSE = chr(60) + "think" + chr(62)
THINK_OPEN = chr(60) + "/" + "think" + chr(62)

GREETING_PREFIXES = (
    "Здравствуйте",
    "Добрый день",
    "Добрый",
    "Привет",
    "Здравствуй",
    "Доброго",
)


def cyr_ratio(text: str) -> float:
    if not text:
        return 0.0
    cyr = sum(1 for char in text if "а" <= char.lower() <= "я" or char in "ёЁ")
    return cyr / len(text)


def is_client_start(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 20:
        return False
    return any(stripped.startswith(prefix) for prefix in GREETING_PREFIXES)


def split_reasoning_and_text(raw: str) -> tuple[str, str]:
    """
    Split model output into reasoning block and client message.

    Args:
        raw: Full model output that may contain English planning before Russian text.

    Returns:
        Tuple of reasoning text and client message text.
    """
    if THINK_CLOSE in raw:
        pre, post = raw.split(THINK_CLOSE, 1)
        pre = pre.replace(THINK_OPEN, "").strip()
        return pre, post.strip()

    lines = raw.split("\n")
    candidates = [index for index, line in enumerate(lines) if is_client_start(line)]
    for index in reversed(candidates):
        tail = "\n".join(lines[index:]).strip()
        if len(tail) >= 80 and cyr_ratio(tail) > 0.45:
            head = "\n".join(lines[:index]).strip()
            return head, tail

    blocks = re.split(r"\n\n+", raw)
    start_index = len(blocks)
    for index in range(len(blocks) - 1, -1, -1):
        block = blocks[index].strip().strip("`")
        if cyr_ratio(block) > 0.55 and len(block) >= 100:
            if not block.lower().startswith("let me"):
                start_index = index
                break
    if start_index < len(blocks):
        reasoning = "\n\n".join(blocks[:start_index]).strip()
        text = "\n\n".join(blocks[start_index:]).strip().strip("`")
        return reasoning, text
    return raw.strip(), ""

