from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from .constants import MONTH_LABELS_UA
from .models import RenderCancelledError
from .settings import build_settings
from .video import render_video


def launch_ui(defaults: argparse.Namespace) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Tkinter is required for --ui mode.") from exc

    root = tk.Tk()
    root.title("Tithing Video Maker")
    root.geometry("700x360")
    root.minsize(700, 360)

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    frame = ttk.Frame(root, padding=14)
    frame.grid(row=0, column=0, sticky="nsew")
    frame.columnconfigure(1, weight=1)
    frame.columnconfigure(2, weight=0)

    target_default = "" if defaults.target is None else str(defaults.target)
    collected_default = "" if defaults.collected is None else str(defaults.collected)
    month_default = defaults.month if defaults.month in MONTH_LABELS_UA else datetime.now().month
    month_choices = [MONTH_LABELS_UA[i] for i in range(1, 13)]
    month_lookup = {label: number for number, label in MONTH_LABELS_UA.items()}
    background_default = "" if defaults.background is None else str(defaults.background)
    output_default = str(defaults.output or Path("output/tithing_video.mp4"))
    qr_default = str(defaults.qr or Path("assets/qr.svg"))
    fps_default = str(defaults.fps or 60)

    target_var = tk.StringVar(value=target_default)
    collected_var = tk.StringVar(value=collected_default)
    month_var = tk.StringVar(value=MONTH_LABELS_UA[month_default])
    background_var = tk.StringVar(value=background_default)
    output_var = tk.StringVar(value=output_default)
    qr_var = tk.StringVar(value=qr_default)
    fps_var = tk.StringVar(value=fps_default)
    status_var = tk.StringVar(value="Fill fields and press Render.")
    progress_var = tk.DoubleVar(value=0.0)
    progress_text_var = tk.StringVar(value="frame_index: 0/0")
    progress_updates: queue.Queue[tuple[int, int]] = queue.Queue()

    summary_frame = ttk.Frame(frame)
    summary_frame.grid(row=0, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 6))
    ttk.Label(summary_frame, text="Collected").grid(row=0, column=0, sticky="w")
    ttk.Entry(summary_frame, textvariable=collected_var, width=12).grid(row=0, column=1, padx=(6, 0))
    ttk.Label(summary_frame, text="From").grid(row=0, column=2, sticky="w", padx=8)
    ttk.Entry(summary_frame, textvariable=target_var, width=12).grid(row=0, column=3)
    ttk.Label(summary_frame, text="For").grid(row=0, column=4, sticky="w", padx=8)
    month_combo = ttk.Combobox(
        summary_frame,
        textvariable=month_var,
        values=month_choices,
        state="readonly",
        width=14,
    )
    month_combo.grid(row=0, column=5)

    ttk.Separator(frame, orient="horizontal").grid(row=1, column=0, columnspan=3, sticky="ew", padx=4, pady=(2, 10))

    ttk.Label(frame, text="Background video").grid(row=2, column=0, sticky="w", padx=4, pady=4)
    ttk.Entry(frame, textvariable=background_var).grid(row=2, column=1, sticky="ew", padx=4, pady=4)

    def browse_background() -> None:
        chosen = filedialog.askopenfilename(
            title="Select background video",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi"), ("All files", "*.*")],
        )
        if chosen:
            background_var.set(chosen)

    ttk.Button(frame, text="Browse", command=browse_background).grid(row=2, column=2, padx=4, pady=4)

    ttk.Label(frame, text="QR image").grid(row=3, column=0, sticky="w", padx=4, pady=4)
    ttk.Entry(frame, textvariable=qr_var).grid(row=3, column=1, sticky="ew", padx=4, pady=4)

    def browse_qr() -> None:
        chosen = filedialog.askopenfilename(
            title="Select QR file",
            filetypes=[("Image files", "*.svg *.png *.jpg *.jpeg"), ("All files", "*.*")],
        )
        if chosen:
            qr_var.set(chosen)

    ttk.Button(frame, text="Browse", command=browse_qr).grid(row=3, column=2, padx=4, pady=4)

    ttk.Label(frame, text="Output video").grid(row=4, column=0, sticky="w", padx=4, pady=4)
    ttk.Entry(frame, textvariable=output_var).grid(row=4, column=1, sticky="ew", padx=4, pady=4)

    def browse_output() -> None:
        chosen = filedialog.asksaveasfilename(
            title="Select output file",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
        )
        if chosen:
            output_var.set(chosen)

    ttk.Button(frame, text="Browse", command=browse_output).grid(row=4, column=2, padx=4, pady=4)

    ttk.Label(frame, text="FPS").grid(row=5, column=0, sticky="w", padx=4, pady=4)
    ttk.Entry(frame, textvariable=fps_var).grid(row=5, column=1, sticky="ew", padx=4, pady=4)

    ttk.Separator(frame, orient="horizontal").grid(row=6, column=0, columnspan=3, sticky="ew", padx=4, pady=(8, 10))

    result_state: dict[str, Path | None] = {"path": None}

    def open_result_path() -> None:
        output_path = result_state["path"]
        if output_path is None:
            messagebox.showwarning("Result not available", "No rendered output is bound to this button yet.")
            return
        if not output_path.exists():
            messagebox.showwarning("Result not found", f"Rendered file does not exist yet:\n{output_path}")
            return

        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", "/select,", str(output_path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(output_path)])
            else:
                subprocess.Popen(["xdg-open", str(output_path.parent)])
        except Exception:
            try:
                if sys.platform.startswith("win"):
                    os.startfile(str(output_path.parent))  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(output_path.parent)])
                else:
                    subprocess.Popen(["xdg-open", str(output_path.parent)])
            except Exception as exc:
                messagebox.showerror("Open failed", f"Could not open result path:\n{exc}")

    open_result_button = ttk.Button(frame, text="Open Result", command=open_result_path, state="disabled")

    status_label = ttk.Label(frame, textvariable=status_var)
    status_label.grid(row=7, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 8))

    progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate", variable=progress_var, maximum=100.0)
    progress.grid(row=8, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 4))
    open_result_button.grid(row=8, column=2, sticky="e", padx=4, pady=(0, 4))
    progress_text = ttk.Label(frame, textvariable=progress_text_var)
    progress_text.grid(row=9, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 4))

    render_button = ttk.Button(frame, text="Render")
    render_button.grid(row=10, column=1, sticky="e", padx=4, pady=4)
    cancel_button = tk.Button(
        frame,
        text="Cancel",
        bg="#d7ce82",
        fg="#3e3922",
        activebackground="#c8be6f",
        activeforeground="#2e2a18",
        relief="flat",
        state="disabled",
        padx=10,
    )
    cancel_button.grid(row=10, column=0, sticky="w", padx=4, pady=4)
    ttk.Button(frame, text="Close", command=root.destroy).grid(row=10, column=2, sticky="e", padx=4, pady=4)

    running = {"active": False}
    cancel_event = threading.Event()

    def sync_open_result_button() -> None:
        has_bound_result = result_state["path"] is not None
        open_result_button.config(state=("normal" if has_bound_result and not running["active"] else "disabled"))

    def set_running(active: bool) -> None:
        running["active"] = active
        render_button.config(state=("disabled" if active else "normal"))
        cancel_button.config(state=("normal" if active else "disabled"))
        sync_open_result_button()
        status_label.update_idletasks()

    def apply_progress(frame_index: int, total_frames: int) -> None:
        total = max(0, int(total_frames))
        current = max(0, int(frame_index))
        if total > 0:
            current = min(current, total)
            progress_var.set((current / total) * 100.0)
            progress_text_var.set(f"frame_index: {current}/{total}")
        else:
            progress_var.set(0.0)
            progress_text_var.set(f"frame_index: {current}/?")

    def flush_progress_queue() -> None:
        latest: tuple[int, int] | None = None
        while True:
            try:
                latest = progress_updates.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            apply_progress(*latest)
        root.after(80, flush_progress_queue)

    root.after(80, flush_progress_queue)

    def on_render() -> None:
        if running["active"]:
            return
        try:
            settings = build_settings(
                target=float(target_var.get()),
                collected=float(collected_var.get()),
                month=month_lookup[month_var.get()],
                background_path=Path(background_var.get()).expanduser(),
                output_path=Path(output_var.get()).expanduser(),
                qr_path=Path(qr_var.get()).expanduser(),
                fps=int(fps_var.get()),
                codec=str(defaults.codec),
                preset=str(defaults.preset),
                threads=defaults.threads,
            )
        except (ValueError, FileNotFoundError) as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("Invalid input", f"Could not parse form values: {exc}")
            return

        result_state["path"] = settings.output_path
        set_running(True)
        cancel_event.clear()
        status_var.set("Rendering video, please wait...")
        apply_progress(0, 0)

        def on_progress(frame_index: int, total_frames: int) -> None:
            progress_updates.put((frame_index, total_frames))

        def worker() -> None:
            error: Exception | None = None
            canceled = False
            try:
                render_video(settings, progress_callback=on_progress, cancel_event=cancel_event)
            except RenderCancelledError:
                canceled = True
                try:
                    settings.output_path.unlink(missing_ok=True)
                except Exception:
                    pass
            except Exception as exc:  # pragma: no cover
                error = exc

            def finish() -> None:
                set_running(False)
                if canceled:
                    status_var.set("Render canceled.")
                    messagebox.showinfo("Canceled", "Render was canceled.")
                elif error is None:
                    apply_progress(1, 1)
                    status_var.set(f"Done: {settings.output_path}")
                    messagebox.showinfo("Done", f"Video rendered:\n{settings.output_path}")
                else:
                    status_var.set("Render failed.")
                    messagebox.showerror("Render failed", str(error))

            root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def on_cancel() -> None:
        if not running["active"] or cancel_event.is_set():
            return
        cancel_event.set()
        cancel_button.config(state="disabled")
        status_var.set("Cancel requested...")

    render_button.config(command=on_render)
    cancel_button.config(command=on_cancel)
    root.mainloop()
