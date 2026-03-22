from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from .models import RenderSettings
from .paths import resolve_existing_path


def build_settings(
    *,
    target: float,
    collected: float,
    month: int,
    background_path: Path,
    output_path: Path,
    qr_path: Path,
    fps: int,
    codec: str = "auto",
    preset: str = "medium",
    threads: int | None = None,
) -> RenderSettings:
    background_path = resolve_existing_path(background_path)
    qr_path = resolve_existing_path(qr_path)

    if target <= 0:
        raise ValueError("--target must be greater than 0.")
    if collected < 0:
        raise ValueError("--collected must be non-negative.")
    if month < 1 or month > 12:
        raise ValueError("--month must be an integer in range 1..12.")
    if fps <= 0:
        raise ValueError("--fps must be greater than 0.")
    if threads is not None and threads <= 0:
        raise ValueError("--threads must be greater than 0 when provided.")
    if not background_path.exists():
        raise FileNotFoundError(f"Background video does not exist: {background_path}")
    if not qr_path.exists():
        raise FileNotFoundError(f"QR file does not exist: {qr_path}")

    return RenderSettings(
        target=target,
        collected=collected,
        month=month,
        background_path=background_path,
        output_path=output_path,
        qr_path=qr_path,
        fps=fps,
        codec=codec,
        preset=preset,
        threads=threads,
    )


def parse_args(args: Iterable[str] | None = None) -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(description="Generate an animated tithing video over a background clip.")
    parser.add_argument("--ui", action="store_true", help="Open a simple windowed mode for entering input values.")
    parser.add_argument("-t", "--target", type=float, help="Target amount for the month.")
    parser.add_argument("-c", "--collected", type=float, help="Collected amount so far.")
    parser.add_argument("-m", "--month", type=int, help="Month number in range 1..12.")
    parser.add_argument(
        "-b",
        "--background",
        type=Path,
        default=Path("backgrounds/background.mp4"),
        help="Path to the loopable background video.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/tithing_video.mp4"),
        help="Output video path (mp4).",
    )
    parser.add_argument(
        "-q",
        "--qr",
        type=Path,
        default=Path("assets/qr.svg"),
        help="Path to static QR image. File must exist.",
    )
    parser.add_argument("-f", "--fps", type=int, default=60, help="Output frames per second.")
    parser.add_argument("--threads", type=int, help="FFmpeg worker threads (default: balanced automatic selection).")
    parser.add_argument(
        "--codec",
        type=str,
        default="auto",
        help="FFmpeg video codec. Use auto to prefer NVIDIA NVENC when available.",
    )
    parser.add_argument("--preset", type=str, default="medium", help="FFmpeg encoding preset.")

    ns = parser.parse_args(args=args)
    return ns, parser


def settings_from_namespace(ns: argparse.Namespace, parser: argparse.ArgumentParser) -> RenderSettings:
    missing: list[str] = []
    if ns.target is None:
        missing.append("--target/-t")
    if ns.collected is None:
        missing.append("--collected/-c")
    if ns.month is None:
        missing.append("--month/-m")
    if missing:
        parser.error(f"Missing required arguments (unless --ui is used): {', '.join(missing)}")

    try:
        return build_settings(
            target=float(ns.target),
            collected=float(ns.collected),
            month=int(ns.month),
            background_path=Path(ns.background),
            output_path=Path(ns.output),
            qr_path=Path(ns.qr),
            fps=int(ns.fps),
            codec=str(ns.codec),
            preset=str(ns.preset),
            threads=None if ns.threads is None else int(ns.threads),
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
        raise AssertionError("Unreachable")
