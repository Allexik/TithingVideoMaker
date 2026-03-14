from __future__ import annotations

from .settings import parse_args, settings_from_namespace
from .ui import launch_ui
from .video import render_video


def main() -> None:
    ns, parser = parse_args()
    if ns.ui:
        launch_ui(ns)
        return
    settings = settings_from_namespace(ns, parser)
    render_video(settings)
