"""Convert the curated examples' known faces at the QQ send boundary.

The host's ``send_service.before_send`` payload stores components under
``message["raw_message"]``. A native QQ face must use the host's ``dict``
component wrapper; a bare ``face`` component would lose its type during host
deserialization. SnowLuma's existing encoder understands this wrapper.

IDs are checked against the installed SnowLuma adapter's ``qq_face_map.py``:
264=捂脸, 111=可怜, 182=笑哭, 9=大哭, 24=饥饿. The WeChat label
``[破涕为笑]`` maps to QQ's semantically similar ``笑哭``; ``😋`` maps to
24, matching that adapter's own ``QQ_FACE_EMOJIS`` table. This is a small
independent mapping, not a copy of emoji2face's implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List

import re


_FACE_IDS: Dict[str, str] = {
    "[捂脸]": "264",
    "[可怜]": "111",
    "[破涕为笑]": "182",
    "[大哭]": "9",
    "😋": "24",
}
_FACE_PATTERN = re.compile(
    "|".join(re.escape(token) for token in _FACE_IDS if token != "😋")
    + "|😋[\ufe0e\ufe0f]?"
)


def convert_qq_segments(segments: List[Dict[str, Any]], platform: str) -> List[Dict[str, Any]]:
    """Split known faces out of QQ text without mutating the original payload.

    Unknown emoji, non-text components, component order, and text-component
    metadata are preserved. Existing face components remain untouched, so
    applying this converter twice does not convert or duplicate them again.
    Non-QQ messages are returned unchanged.
    """

    if platform.strip().lower() != "qq":
        return segments

    result: List[Dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict) or segment.get("type") != "text":
            result.append(segment)
            continue

        text = segment.get("data")
        if not isinstance(text, str):
            result.append(segment)
            continue

        matches = list(_FACE_PATTERN.finditer(text))
        if not matches:
            result.append(segment)
            continue

        cursor = 0
        for match in matches:
            if match.start() > cursor:
                result.append({**segment, "data": text[cursor:match.start()]})
            token = match.group(0).rstrip("\ufe0e\ufe0f")
            result.append(
                {"type": "dict", "data": {"type": "face", "data": {"id": _FACE_IDS[token]}}}
            )
            cursor = match.end()
        if cursor < len(text):
            result.append({**segment, "data": text[cursor:]})

    return result
