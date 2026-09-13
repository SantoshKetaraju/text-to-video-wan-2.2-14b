from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from watermark import WATERMARK_TEXT, WatermarkError, apply_disclosure_watermark


class WatermarkTests(unittest.TestCase):
    def test_success_replaces_source_only_after_ffmpeg_output_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "output.mp4"
            font = root / "font.ttf"
            source.write_bytes(b"raw-video")
            font.write_bytes(b"font")

            def fake_run(command, **_kwargs):
                Path(command[-1]).write_bytes(b"watermarked-video")
                return SimpleNamespace(returncode=0, stderr="", stdout="")

            with (
                patch.dict(os.environ, {"WATERMARK_FONT_PATH": str(font)}),
                patch("watermark.shutil.which", return_value="/usr/bin/ffmpeg"),
                patch("watermark.subprocess.run", side_effect=fake_run) as run,
            ):
                apply_disclosure_watermark(source)

            self.assertEqual(source.read_bytes(), b"watermarked-video")
            command = run.call_args.args[0]
            self.assertIn("-vf", command)
            video_filter = command[command.index("-vf") + 1]
            self.assertIn(WATERMARK_TEXT, video_filter)
            self.assertIn("fontcolor=white@0.45", video_filter)
            self.assertIn("fontsize=h/28", video_filter)
            self.assertIn("x=w-tw-18:y=h-th-18", video_filter)

    def test_failure_preserves_original_and_removes_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "output.mp4"
            font = root / "font.ttf"
            source.write_bytes(b"raw-video")
            font.write_bytes(b"font")

            def fake_run(command, **_kwargs):
                Path(command[-1]).write_bytes(b"partial")
                return SimpleNamespace(
                    returncode=1,
                    stderr="test failure",
                    stdout="",
                )

            with (
                patch.dict(os.environ, {"WATERMARK_FONT_PATH": str(font)}),
                patch("watermark.shutil.which", return_value="/usr/bin/ffmpeg"),
                patch("watermark.subprocess.run", side_effect=fake_run),
            ):
                with self.assertRaisesRegex(WatermarkError, "test failure"):
                    apply_disclosure_watermark(source)

            self.assertEqual(source.read_bytes(), b"raw-video")
            self.assertFalse((root / ".output.watermark.mp4").exists())


if __name__ == "__main__":
    unittest.main()
