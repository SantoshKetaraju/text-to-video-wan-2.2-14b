from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


WATERMARK_TEXT = "AI modified"
DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


class WatermarkError(RuntimeError):
    pass


def _escape_drawtext_value(value: str) -> str:
    return (
        value.replace("\\", "/")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(",", "\\,")
    )


def apply_disclosure_watermark(video_path: str | Path) -> None:
    """Apply the disclosure watermark to an MP4 in place.

    FFmpeg writes a separate temporary MP4 and the source is replaced only
    after a successful render, so a partial file can never be uploaded.
    """

    source_path = Path(video_path)
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        raise WatermarkError("Cannot watermark a missing or empty video")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise WatermarkError("FFmpeg is not available")

    font_path = Path(os.environ.get("WATERMARK_FONT_PATH", DEFAULT_FONT_PATH))
    if not font_path.is_file():
        raise WatermarkError(f"Watermark font is missing: {font_path}")

    escaped_font_path = _escape_drawtext_value(str(font_path.resolve()))
    escaped_text = _escape_drawtext_value(WATERMARK_TEXT)
    video_filter = (
        f"drawtext=fontfile='{escaped_font_path}':"
        f"text='{escaped_text}':"
        "fontcolor=white@0.45:fontsize=h/28:"
        "box=0:x=w-tw-18:y=h-th-18"
    )
    temporary_path = source_path.with_name(
        f".{source_path.stem}.watermark{source_path.suffix}"
    )
    temporary_path.unlink(missing_ok=True)

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-map_metadata",
        "-1",
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise WatermarkError(
                f"FFmpeg watermarking failed: {detail or 'unknown error'}"
            )
        if not temporary_path.is_file() or temporary_path.stat().st_size <= 0:
            raise WatermarkError("FFmpeg produced an empty watermarked video")
        os.replace(temporary_path, source_path)
    finally:
        temporary_path.unlink(missing_ok=True)
