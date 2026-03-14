from __future__ import annotations

import os
import threading
from typing import Callable

import numpy as np

try:
    from moviepy.editor import VideoClip, VideoFileClip
except ImportError:
    from moviepy import VideoClip, VideoFileClip

try:
    from proglog import ProgressBarLogger
except ImportError:  # pragma: no cover
    ProgressBarLogger = None

from .models import RenderCancelledError, RenderSettings
from .renderer import SceneRenderer


def render_video(
    settings: RenderSettings,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    effective_threads = settings.threads if settings.threads is not None else max(1, os.cpu_count() or 1)

    background_clip = VideoFileClip(str(settings.background_path))
    output_clip: VideoClip | None = None
    try:
        duration = float(background_clip.duration)
        width, height = background_clip.size
        renderer = SceneRenderer(width=width, height=height, settings=settings)

        def frame_fn(t: float) -> np.ndarray:
            if cancel_event is not None and cancel_event.is_set():
                raise RenderCancelledError("Render canceled by user.")
            background_frame = background_clip.get_frame(t)
            return renderer.render_frame(background_frame, t_seconds=t, duration=duration)

        output_clip = VideoClip(frame_fn, duration=duration)
        if background_clip.audio is not None:
            audio = background_clip.audio
            if hasattr(audio, "subclip"):
                audio = audio.subclip(0, duration)
            elif hasattr(audio, "subclipped"):
                audio = audio.subclipped(0, duration)

            if hasattr(output_clip, "set_audio"):
                output_clip = output_clip.set_audio(audio)
            else:
                output_clip = output_clip.with_audio(audio)

        logger: object = "bar"
        if progress_callback is not None and ProgressBarLogger is not None:

            class FrameIndexLogger(ProgressBarLogger):
                def __init__(self, callback: Callable[[int, int], None]):
                    super().__init__()
                    self._callback = callback
                    self._total_frames = 0

                def bars_callback(self, bar: str, attr: str, value: int, old_value: int | None = None) -> None:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RenderCancelledError("Render canceled by user.")
                    if bar != "frame_index":
                        return
                    if attr == "total":
                        self._total_frames = int(value)
                        self._callback(0, self._total_frames)
                        return
                    if attr == "index":
                        self._callback(int(value), self._total_frames)

            logger = FrameIndexLogger(progress_callback)

        output_clip.write_videofile(
            str(settings.output_path),
            fps=settings.fps,
            codec=settings.codec,
            audio_codec="aac",
            threads=effective_threads,
            preset=settings.preset,
            logger=logger,
        )
    finally:
        if output_clip is not None:
            output_clip.close()
        background_clip.close()
