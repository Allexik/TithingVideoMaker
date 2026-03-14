from __future__ import annotations

import functools
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import RenderSettings

_AUTO_CODECS = {"", "auto", "default"}
_MP4_SUFFIXES = {".m4v", ".mov", ".mp4"}
_HARDWARE_CODEC_SUFFIXES = ("_amf", "_d3d12va", "_mf", "_nvenc", "_qsv", "_vaapi")
_NVENC_PRESET_ALIASES = {
    "fast": "p3",
    "faster": "p2",
    "medium": "p4",
    "slower": "p6",
    "slow": "p5",
    "superfast": "p2",
    "ultrafast": "p1",
    "veryfast": "p2",
    "veryslow": "p7",
}
_PIXEL_FORMAT_CODECS = {
    "av1_nvenc",
    "h264_amf",
    "h264_mf",
    "h264_nvenc",
    "h264_qsv",
    "hevc_amf",
    "hevc_d3d12va",
    "hevc_mf",
    "hevc_nvenc",
    "hevc_qsv",
    "libx264",
    "libx265",
}


@dataclass(frozen=True)
class EncodePlan:
    codec: str
    preset: str
    threads: int
    ffmpeg_params: tuple[str, ...]
    pixel_format: str | None


def build_encode_plan(settings: RenderSettings) -> EncodePlan:
    requested_codec = (settings.codec or "").strip().lower()
    codec = requested_codec
    if requested_codec in _AUTO_CODECS:
        codec = "h264_nvenc" if can_use_encoder("h264_nvenc") else "libx264"
    elif is_hardware_codec(requested_codec) and not can_use_encoder(requested_codec):
        raise RuntimeError(
            f"Requested codec '{settings.codec}' is not available in this FFmpeg/runtime setup. "
            "Use --codec auto or a software codec such as libx264."
        )

    preset = normalize_preset(codec=codec, preset=settings.preset)
    threads = settings.threads if settings.threads is not None else recommended_threads(codec)
    ffmpeg_params = list(default_ffmpeg_params(codec=codec, output_path=settings.output_path))
    pixel_format = "yuv420p" if codec in _PIXEL_FORMAT_CODECS else None
    return EncodePlan(
        codec=codec,
        preset=preset,
        threads=max(1, threads),
        ffmpeg_params=tuple(ffmpeg_params),
        pixel_format=pixel_format,
    )


def normalize_preset(codec: str, preset: str) -> str:
    chosen = (preset or "").strip()
    if not chosen:
        return "p4" if codec.endswith("_nvenc") else "medium"
    if codec.endswith("_nvenc"):
        return _NVENC_PRESET_ALIASES.get(chosen.lower(), chosen)
    return chosen


def recommended_threads(codec: str) -> int:
    cpu_count = max(1, os.cpu_count() or 1)
    if is_hardware_codec(codec):
        return min(2, cpu_count)
    return max(1, min(8, cpu_count - 1))


def is_hardware_codec(codec: str) -> bool:
    lowered = codec.lower()
    return lowered.endswith(_HARDWARE_CODEC_SUFFIXES)


def default_ffmpeg_params(codec: str, output_path: Path) -> tuple[str, ...]:
    params: list[str] = []
    if output_path.suffix.lower() in _MP4_SUFFIXES:
        params.extend(["-movflags", "+faststart"])
    if codec.endswith("_nvenc"):
        params.extend(
            [
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "19",
                "-b:v",
                "0",
                "-spatial-aq",
                "1",
                "-temporal-aq",
                "1",
                "-rc-lookahead",
                "20",
            ]
        )
    return tuple(params)


def can_use_encoder(codec: str) -> bool:
    return codec.lower() in available_encoders() and encoder_runtime_available(codec)


@functools.lru_cache(maxsize=1)
def available_encoders() -> frozenset[str]:
    ffmpeg_exe = ffmpeg_executable()
    if ffmpeg_exe is None:
        return frozenset()

    result = subprocess.run(
        [ffmpeg_exe, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return frozenset()

    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1].lower())
    return frozenset(encoders)


@functools.lru_cache(maxsize=16)
def encoder_runtime_available(codec: str) -> bool:
    ffmpeg_exe = ffmpeg_executable()
    if ffmpeg_exe is None:
        return False

    command = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:r=24",
        "-frames:v",
        "1",
        "-c:v",
        codec,
    ]
    if codec.endswith("_nvenc"):
        command.extend(["-preset", "p4"])
    command.extend(["-f", "null", "-"])

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.returncode == 0


@functools.lru_cache(maxsize=1)
def ffmpeg_executable() -> str | None:
    try:
        import imageio_ffmpeg  # type: ignore
    except ImportError:
        imageio_ffmpeg = None

    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

    return shutil.which("ffmpeg")
