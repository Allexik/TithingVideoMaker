from __future__ import annotations

import os
from pathlib import Path
from typing import TypeAlias

from PIL import Image, ImageDraw, ImageFont

from .constants import MONTH_LABELS_UA

Color: TypeAlias = tuple[int, int, int, int]
TextSegment: TypeAlias = tuple[str, ImageFont.ImageFont, Color]
MeasuredTextSegment: TypeAlias = tuple[str, ImageFont.ImageFont, Color, tuple[int, int, int, int], int]


def month_label_ua(month: int) -> str:
    return MONTH_LABELS_UA[month]


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Invalid hex color: {value}")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def ease_out_cubic(value: float) -> float:
    x = clamp(value)
    return 1 - (1 - x) ** 3


def ease_in_out_cubic(value: float) -> float:
    x = clamp(value)
    if x < 0.5:
        return 4 * x**3
    return 1 - ((-2 * x + 2) ** 3) / 2


def segment_progress(t: float, start: float, end: float) -> float:
    if end <= start:
        return 0.0
    return clamp((t - start) / (end - start))


def segment_alpha(
    t: float, start: float, end: float, fade_in_share: float = 0.15, fade_out_share: float = 0.15
) -> float:
    if t < start or t > end:
        return 0.0
    p = segment_progress(t, start, end)
    fade_in = clamp(p / max(fade_in_share, 1e-6))
    fade_out = clamp((1.0 - p) / max(fade_out_share, 1e-6))
    return clamp(min(fade_in, fade_out))


def format_amount(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        rendered = f"{int(round(value)):,}"
    else:
        rendered = f"{value:,.2f}"
    return rendered.replace(",", " ")


def find_font_path(bold: bool = False) -> str | None:
    windows_candidates = [
        str(Path("assets/fonts/Nunito-Bold.ttf")) if bold else str(Path("assets/fonts/Nunito-Regular.ttf")),
        str(Path("assets/fonts/NunitoSans-Bold.ttf"))
        if bold
        else str(Path("assets/fonts/NunitoSans-Regular.ttf")),
        str(Path("assets/fonts/Fallback-Bold.ttf")) if bold else str(Path("assets/fonts/Fallback-Regular.ttf")),
        r"C:\Windows\Fonts\Nunito-Bold.ttf" if bold else r"C:\Windows\Fonts\Nunito-Regular.ttf",
        r"C:\Windows\Fonts\NunitoSans-Bold.ttf" if bold else r"C:\Windows\Fonts\NunitoSans-Regular.ttf",
        r"C:\Windows\Fonts\Nunito-ExtraBold.ttf" if bold else r"C:\Windows\Fonts\Nunito-Light.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\NotoSans-Regular.ttf",
        r"C:\Windows\Fonts\NotoSans-Bold.ttf",
    ]
    unix_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in windows_candidates + unix_candidates:
        if os.path.exists(path):
            return path
    return None


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = find_font_path(bold=bold)
    if font_path:
        return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def find_heart_font_path() -> str | None:
    candidates = [
        str(Path("assets/fonts/HeartSymbol.ttf")),
        str(Path("assets/fonts/SegoeUISymbol.ttf")),
        r"C:\Windows\Fonts\seguisym.ttf",
        r"C:\Windows\Fonts\seguiemj.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return find_font_path(bold=True) or find_font_path(bold=False)


def load_heart_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = find_heart_font_path()
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    wrapped_lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            wrapped_lines.append("")
            continue
        line_words: list[str] = []
        for word in words:
            candidate = " ".join(line_words + [word])
            left, top, right, bottom = draw.textbbox((0, 0), candidate, font=font)
            width = right - left
            if width <= max_width or not line_words:
                line_words.append(word)
                continue
            wrapped_lines.append(" ".join(line_words))
            line_words = [word]
        if line_words:
            wrapped_lines.append(" ".join(line_words))
    return "\n".join(wrapped_lines)


def draw_centered_multiline_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: int,
    center_y: int,
    font: ImageFont.ImageFont,
    fill: Color,
    spacing: int = 8,
    shadow_alpha: int = 120,
) -> tuple[int, int]:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=spacing)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = center_x - width // 2 - bbox[0]
    y = center_y - height // 2 - bbox[1]
    shadow = (0, 0, 0, min(255, shadow_alpha))
    draw.multiline_text((x + 2, y + 2), text, font=font, fill=shadow, align="center", spacing=spacing)
    draw.multiline_text((x, y), text, font=font, fill=fill, align="center", spacing=spacing)
    return width, height


def measure_text_segments(
    draw: ImageDraw.ImageDraw,
    segments: list[TextSegment],
) -> tuple[int, int, int, list[MeasuredTextSegment]]:
    items: list[MeasuredTextSegment] = []
    min_top = 0
    max_bottom = 0
    total_width = 0
    first = True
    for text, font, fill in segments:
        if not text:
            continue
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        total_width += width
        if first:
            min_top = bbox[1]
            max_bottom = bbox[3]
            first = False
        else:
            min_top = min(min_top, bbox[1])
            max_bottom = max(max_bottom, bbox[3])
        items.append((text, font, fill, bbox, width))

    height = 0 if first else (max_bottom - min_top)
    return total_width, height, min_top, items


def draw_centered_segments(
    draw: ImageDraw.ImageDraw,
    segments: list[TextSegment],
    center_x: int,
    center_y: int,
    shadow_alpha: int = 95,
) -> tuple[int, int]:
    total_width, height, min_top, items = measure_text_segments(draw, segments)
    if not items:
        return 0, 0

    left = center_x - total_width // 2
    baseline_y = center_y - height // 2 - min_top
    shadow = (0, 0, 0, min(255, shadow_alpha))

    cursor = left
    for text, font, fill, bbox, width in items:
        x = cursor - bbox[0]
        draw.text((x + 2, baseline_y + 2), text, font=font, fill=shadow)
        draw.text((x, baseline_y), text, font=font, fill=fill)
        cursor += width
    return total_width, height


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: int,
    center_y: int,
    font: ImageFont.ImageFont,
    fill: Color,
    shadow_alpha: int = 100,
) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = center_x - width // 2 - bbox[0]
    y = center_y - height // 2 - bbox[1]
    shadow = (0, 0, 0, min(255, shadow_alpha))
    draw.text((x + 2, y + 2), text, font=font, fill=shadow)
    draw.text((x, y), text, font=font, fill=fill)
    return width, height


def apply_layer_alpha(layer: Image.Image, alpha: float) -> Image.Image:
    factor = clamp(alpha)
    if factor >= 0.999:
        return layer
    alpha_channel = layer.getchannel("A").point(lambda px: int(px * factor))
    layer.putalpha(alpha_channel)
    return layer
