from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from tithing_video_maker.encoding import build_encode_plan, normalize_preset, recommended_threads
from tithing_video_maker.models import RenderSettings


def make_settings(codec: str = "auto", preset: str = "medium", threads: int | None = None) -> RenderSettings:
    return RenderSettings(
        target=50000,
        collected=32750,
        month=3,
        background_path=Path("backgrounds/background.mp4"),
        output_path=Path("output/test.mp4"),
        qr_path=Path("assets/qr.svg"),
        fps=60,
        codec=codec,
        preset=preset,
        threads=threads,
    )


class EncodePlanTests(unittest.TestCase):
    @patch("tithing_video_maker.encoding.can_use_encoder", return_value=True)
    def test_auto_prefers_nvenc_when_available(self, _can_use_encoder: object) -> None:
        plan = build_encode_plan(make_settings())
        self.assertEqual(plan.codec, "h264_nvenc")
        self.assertEqual(plan.preset, "p4")
        self.assertEqual(plan.pixel_format, "yuv420p")
        self.assertIn("-rc-lookahead", plan.ffmpeg_params)

    @patch("tithing_video_maker.encoding.can_use_encoder", return_value=False)
    def test_auto_falls_back_to_libx264_when_nvenc_unavailable(self, _can_use_encoder: object) -> None:
        plan = build_encode_plan(make_settings())
        self.assertEqual(plan.codec, "libx264")
        self.assertEqual(plan.preset, "medium")
        self.assertEqual(plan.pixel_format, "yuv420p")
        self.assertIn("+faststart", plan.ffmpeg_params)

    @patch("tithing_video_maker.encoding.can_use_encoder", return_value=False)
    def test_explicit_unavailable_hardware_codec_raises(self, _can_use_encoder: object) -> None:
        with self.assertRaises(RuntimeError):
            build_encode_plan(make_settings(codec="h264_nvenc"))

    def test_nvenc_preset_aliases_map_from_x264_style_names(self) -> None:
        self.assertEqual(normalize_preset("h264_nvenc", "medium"), "p4")
        self.assertEqual(normalize_preset("h264_nvenc", "veryfast"), "p2")
        self.assertEqual(normalize_preset("h264_nvenc", "veryslow"), "p7")

    @patch("os.cpu_count", return_value=12)
    def test_recommended_threads_balance_hardware_and_software_encoders(self, _cpu_count: object) -> None:
        self.assertEqual(recommended_threads("h264_nvenc"), 2)
        self.assertEqual(recommended_threads("libx264"), 8)


if __name__ == "__main__":
    unittest.main()
