from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RenderSettings:
    target: float
    collected: float
    month: int
    background_path: Path
    output_path: Path
    qr_path: Path
    fps: int
    codec: str
    preset: str
    threads: int | None


class RenderCancelledError(Exception):
    pass
