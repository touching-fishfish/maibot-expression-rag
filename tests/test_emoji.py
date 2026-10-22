"""Offline checks for the QQ send-boundary conversion, with no network calls."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest import TestCase

import importlib.util
import unittest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "expression_rag" / "emoji.py"
_SPEC = importlib.util.spec_from_file_location("expression_rag_emoji_test_target", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
convert_qq_segments = _MODULE.convert_qq_segments


def face(face_id: str) -> dict:
    return {"type": "dict", "data": {"type": "face", "data": {"id": face_id}}}


class EmojiConversionTests(TestCase):
    def test_all_five_curated_faces_use_host_dict_wrapper(self) -> None:
        result = convert_qq_segments(
            [{"type": "text", "data": "[捂脸][可怜][破涕为笑][大哭]😋"}], "qq"
        )
        self.assertEqual(result, [face("264"), face("111"), face("182"), face("9"), face("24")])

    def test_preserves_text_order_metadata_and_input(self) -> None:
        original = [{"type": "text", "data": "你看[捂脸]，挺好😋！", "source": "reply"}]
        snapshot = deepcopy(original)
        result = convert_qq_segments(original, "qq")
        self.assertEqual(
            result,
            [
                {"type": "text", "data": "你看", "source": "reply"},
                face("264"),
                {"type": "text", "data": "，挺好", "source": "reply"},
                face("24"),
                {"type": "text", "data": "！", "source": "reply"},
            ],
        )
        self.assertEqual(original, snapshot)

    def test_unknown_faces_unicode_and_non_text_components_are_preserved(self) -> None:
        segments = [
            {"type": "text", "data": "[未知表情]🦄❤️[文件]"},
            {"type": "image", "data": "[捂脸]", "hash": "image-hash"},
            {"type": "at", "data": {"target_user_id": "test-user"}},
            {"type": "reply", "data": "quoted-message"},
            {"type": "dict", "data": {"type": "face", "data": {"id": "264"}}},
        ]
        self.assertEqual(convert_qq_segments(segments, "qq"), segments)

    def test_non_qq_messages_are_unchanged(self) -> None:
        segments = [{"type": "text", "data": "[捂脸]😋"}]
        for platform in ("webui", "wechat", "", "telegram"):
            with self.subTest(platform=platform):
                self.assertEqual(convert_qq_segments(segments, platform), segments)

    def test_conversion_is_idempotent(self) -> None:
        once = convert_qq_segments([{"type": "text", "data": "好[可怜]😋"}], "qq")
        self.assertEqual(convert_qq_segments(once, "qq"), once)

    def test_optional_unicode_variation_selectors_do_not_leave_residue(self) -> None:
        result = convert_qq_segments([{"type": "text", "data": "😋\ufe0f😋\ufe0e"}], "QQ")
        self.assertEqual(result, [face("24"), face("24")])

    def test_empty_or_nonstandard_text_payload_is_preserved(self) -> None:
        segments = [
            {"type": "text", "data": ""},
            {"type": "text", "data": None},
            {"type": "text", "data": {"text": "[捂脸]"}},
        ]
        self.assertEqual(convert_qq_segments(segments, "qq"), segments)


if __name__ == "__main__":
    unittest.main()
