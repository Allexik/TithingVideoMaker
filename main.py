from __future__ import annotations

import argparse
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from moviepy.editor import VideoClip, VideoFileClip
except ImportError:
    from moviepy import VideoClip, VideoFileClip


VERSE_TEXT = (
    'Нехай кожен дає, як серце йому призволяє, не в смутку й не з примусу, '
    'бо Бог любить того, хто з радістю дає!'
)
VERSE_REF = "2 Коринтян 9:7"


@dataclass
class RenderSettings:
    target: float
    collected: float
    month: int
    background_path: Path
    output_path: Path
    qr_path: Path
    fps: int


MONTH_LABELS_UA = {
    1: "Січень",
    2: "Лютий",
    3: "Березень",
    4: "Квітень",
    5: "Травень",
    6: "Червень",
    7: "Липень",
    8: "Серпень",
    9: "Вересень",
    10: "Жовтень",
    11: "Листопад",
    12: "Грудень",
}


SEASON_COLORS_HEX = {
    "winter": "#1f4083",
    "spring": "#218a42",
    "summer": "#8a236d",
    "autumn": "#8c3323",
}


def month_label_ua(month: int) -> str:
    return MONTH_LABELS_UA[month]


def month_season_name(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


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
    fill: tuple[int, int, int, int],
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


def _measure_text_segments(
    draw: ImageDraw.ImageDraw,
    segments: list[tuple[str, ImageFont.ImageFont, tuple[int, int, int, int]]],
) -> tuple[int, int, int, list[tuple[str, ImageFont.ImageFont, tuple[int, int, int, int], tuple[int, int, int, int], int]]]:
    items: list[tuple[str, ImageFont.ImageFont, tuple[int, int, int, int], tuple[int, int, int, int], int]] = []
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
    segments: list[tuple[str, ImageFont.ImageFont, tuple[int, int, int, int]]],
    center_x: int,
    center_y: int,
    shadow_alpha: int = 95,
) -> tuple[int, int]:
    total_width, height, min_top, items = _measure_text_segments(draw, segments)
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
    fill: tuple[int, int, int, int],
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


def _create_placeholder_qr_image(size: int = 1024) -> Image.Image:
    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    border = size // 24
    draw.rectangle(
        (border, border, size - border, size - border), outline=(25, 25, 25, 255), width=size // 40
    )

    cell = size // 8
    for row in range(8):
        for col in range(8):
            if (row + col) % 2 == 0 and (row * col) % 3 != 0:
                x0 = col * cell + border
                y0 = row * cell + border
                x1 = min(size - border, x0 + cell - border // 2)
                y1 = min(size - border, y0 + cell - border // 2)
                draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, 255))

    text_font = load_font(size // 7, bold=True)
    text = "QR"
    text_bbox = draw.textbbox((0, 0), text, font=text_font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    text_pos = ((size - text_w) // 2, (size - text_h) // 2)
    draw.rectangle(
        (
            text_pos[0] - border,
            text_pos[1] - border,
            text_pos[0] + text_w + border,
            text_pos[1] + text_h + border,
        ),
        fill=(255, 255, 255, 220),
    )
    draw.text(text_pos, text, font=text_font, fill=(0, 0, 0, 255))
    return image


def _svg_number(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    text = value.strip()
    match = re.match(r"[-+]?\d*\.?\d+", text)
    if not match:
        return default
    return float(match.group(0))


def _parse_rotate_center(transform: str | None) -> tuple[float, float] | None:
    if not transform:
        return None
    match = re.search(
        r"rotate\(\s*[-+]?\d*\.?\d+\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)",
        transform,
    )
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _looks_like_black_square(image: Image.Image) -> bool:
    arr = np.asarray(image.convert("RGBA"))
    alpha = arr[..., 3] > 0
    if alpha.sum() == 0:
        return False
    rgb = arr[..., :3]
    black = ((rgb == 0).all(axis=-1) & alpha).sum()
    ratio = float(black) / float(alpha.sum())
    unique_colors = len(np.unique(arr.reshape(-1, 4), axis=0))
    return ratio > 0.98 and unique_colors <= 2


def _paint_finder_pattern(matrix: np.ndarray, top: int, left: int) -> None:
    n = matrix.shape[0]
    for y in range(7):
        for x in range(7):
            gy = top + y
            gx = left + x
            if gy < 0 or gx < 0 or gy >= n or gx >= n:
                continue
            is_black = x in (0, 6) or y in (0, 6) or (2 <= x <= 4 and 2 <= y <= 4)
            matrix[gy, gx] = is_black


def _render_stylized_qr_svg_as_grid(path: Path, size: int = 1024) -> Image.Image:
    tree = ET.parse(path)
    root = tree.getroot()

    width = max(1, int(round(_svg_number(root.attrib.get("width"), 500.0))))
    height = max(1, int(round(_svg_number(root.attrib.get("height"), 500.0))))

    dot_clip = root.find(".//{*}clipPath[@id='clip-path-dot-color']")
    if dot_clip is None:
        raise RuntimeError("Unsupported SVG QR format: clip-path-dot-color was not found.")

    module_centers: list[tuple[float, float]] = []
    module_sizes: list[float] = []

    for elem in dot_clip:
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "rect":
            x = _svg_number(elem.attrib.get("x"))
            y = _svg_number(elem.attrib.get("y"))
            w = _svg_number(elem.attrib.get("width"))
            h = _svg_number(elem.attrib.get("height"))
            if w > 0 and h > 0:
                module_centers.append((x + w / 2.0, y + h / 2.0))
                module_sizes.extend([w, h])
            continue

        if tag == "circle":
            cx = _svg_number(elem.attrib.get("cx"))
            cy = _svg_number(elem.attrib.get("cy"))
            r = _svg_number(elem.attrib.get("r"))
            module_centers.append((cx, cy))
            if r > 0:
                module_sizes.append(r * 2.0)
            continue

        if tag == "path":
            center = _parse_rotate_center(elem.attrib.get("transform"))
            if center is not None:
                module_centers.append(center)
                continue

            d_attr = elem.attrib.get("d", "")
            match = re.search(r"M\s*([-+]?\d*\.?\d+)\s*([-+]?\d*\.?\d+)", d_attr)
            if match:
                cell = module_sizes[0] if module_sizes else 14.0
                x = float(match.group(1)) + cell / 2.0
                y = float(match.group(2)) + cell / 2.0
                module_centers.append((x, y))

    if not module_centers:
        raise RuntimeError("No QR modules could be extracted from SVG.")

    xs = [c[0] for c in module_centers]
    ys = [c[1] for c in module_centers]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    step = float(np.median(module_sizes)) if module_sizes else 14.0
    if step <= 0:
        step = 14.0

    cols = int(round((max_x - min_x) / step)) + 1
    rows = int(round((max_y - min_y) / step)) + 1
    grid_size = max(cols, rows)
    matrix = np.zeros((grid_size, grid_size), dtype=bool)

    for cx, cy in module_centers:
        gx = int(round((cx - min_x) / step))
        gy = int(round((cy - min_y) / step))
        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            matrix[gy, gx] = True

    # Ensure finder patterns exist even when stylized clips omit them.
    _paint_finder_pattern(matrix, top=0, left=0)
    _paint_finder_pattern(matrix, top=0, left=grid_size - 7)
    _paint_finder_pattern(matrix, top=grid_size - 7, left=0)

    img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    origin_x = min_x - step / 2.0
    origin_y = min_y - step / 2.0

    for gy in range(grid_size):
        for gx in range(grid_size):
            if not matrix[gy, gx]:
                continue
            x0 = int(round(origin_x + gx * step))
            y0 = int(round(origin_y + gy * step))
            x1 = int(round(x0 + step))
            y1 = int(round(y0 + step))
            draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, 255))

    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def _load_svg_qr(path: Path, size: int = 1024) -> Image.Image:
    image: Image.Image | None = None
    try:
        import fitz  # type: ignore

        with fitz.open(str(path)) as doc:
            if doc.page_count >= 1:
                page = doc[0]
                rect = page.rect
                if rect.width > 0 and rect.height > 0:
                    scale = size / max(rect.width, rect.height)
                    matrix = fitz.Matrix(scale, scale)
                    pix = page.get_pixmap(matrix=matrix, alpha=True)
                    image = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
    except Exception:
        image = None

    if image is None or _looks_like_black_square(image):
        image = _render_stylized_qr_svg_as_grid(path, size=size)

    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def ensure_placeholder_qr(path: Path, size: int = 1024) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"QR file does not exist: {path}")
    if path.suffix.lower() == ".svg":
        return _load_svg_qr(path, size=size)
    return Image.open(path).convert("RGBA")


class SceneRenderer:
    def __init__(self, width: int, height: int, settings: RenderSettings):
        self.width = width
        self.height = height
        self.settings = settings
        self.month_label = month_label_ua(settings.month)
        self.season_name = month_season_name(settings.month)
        self.season_color = hex_to_rgb(SEASON_COLORS_HEX[self.season_name])
        self.ratio = 0.0 if settings.target <= 0 else settings.collected / settings.target
        self.qr_image = ensure_placeholder_qr(settings.qr_path)

        self.font_verse = load_font(max(36, int(height * 0.074)), bold=True)
        self.font_reference = load_font(max(30, int(height * 0.049)), bold=True)
        self.font_label = load_font(max(38, int(height * 0.070)), bold=True)
        self.font_number = load_font(max(34, int(height * 0.060)), bold=True)
        self.font_percent = load_font(max(50, int(height * 0.097)), bold=True)
        self.font_donate = load_font(max(34, int(height * 0.060)), bold=True)
        self.font_heart = load_heart_font(max(42, int(height * 0.074)))

    def _draw_verse_section(self, overlay: Image.Image, t_norm: float) -> None:
        start = 0.03
        end = 0.36
        alpha = segment_alpha(t_norm, start, end, fade_in_share=0.18, fade_out_share=0.13)
        if alpha <= 0:
            return

        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        max_text_width = int(self.width * 0.84)
        verse_wrapped = wrap_text(draw, VERSE_TEXT, self.font_verse, max_text_width)

        center_x = self.width // 2
        verse_y = int(self.height * 0.43)
        verse_size = draw_centered_multiline_text(
            draw,
            verse_wrapped,
            center_x=center_x,
            center_y=verse_y,
            font=self.font_verse,
            fill=(255, 255, 255, 255),
            spacing=max(10, int(self.height * 0.012)),
            shadow_alpha=120,
        )

        gap = int(self.height * 0.13)
        reference_y = verse_y + verse_size[1] // 2 + gap
        draw_centered_multiline_text(
            draw,
            VERSE_REF,
            center_x=center_x,
            center_y=reference_y,
            font=self.font_reference,
            fill=(245, 245, 245, 240),
            spacing=4,
            shadow_alpha=110,
        )

        overlay.alpha_composite(apply_layer_alpha(layer, alpha))

    def _draw_donut(
        self,
        draw: ImageDraw.ImageDraw,
        center: tuple[int, int],
        radius: int,
        thickness: int,
        progress_ratio: float,
        alpha: float,
    ) -> None:
        cx, cy = center
        x0, y0 = cx - radius, cy - radius
        x1, y1 = cx + radius, cy + radius

        base_color = (255, 255, 255, int(110 * alpha))
        draw.ellipse((x0, y0, x1, y1), outline=base_color, width=thickness)

        main_ratio = max(0.0, min(1.0, progress_ratio))
        if main_ratio > 0:
            sr, sg, sb = self.season_color
            draw.arc(
                (x0, y0, x1, y1),
                start=-90,
                end=-90 + 360 * main_ratio,
                fill=(sr, sg, sb, int(255 * alpha)),
                width=thickness,
            )

        over_ratio = max(0.0, progress_ratio - 1.0)
        if over_ratio > 0:
            outer_pad = int(thickness * 0.7)
            ox0, oy0 = x0 - outer_pad, y0 - outer_pad
            ox1, oy1 = x1 + outer_pad, y1 + outer_pad
            draw.arc(
                (ox0, oy0, ox1, oy1),
                start=-90,
                end=-90 + 360 * min(over_ratio, 1.0),
                fill=(253, 124, 124, int(255 * alpha)),
                width=max(2, thickness // 2),
            )

    def _draw_money_section(self, overlay: Image.Image, t_norm: float) -> None:
        start = 0.42
        end = 0.94
        section_alpha = segment_alpha(t_norm, start, end, fade_in_share=0.10, fade_out_share=0.12)
        if section_alpha <= 0:
            return

        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        enter = ease_out_cubic(segment_progress(t_norm, start + 0.01, start + 0.08))
        chart_anim = ease_out_cubic(segment_progress(t_norm, start + 0.02, start + 0.09))

        pair_center_x = self.width // 2
        pair_offset = int(self.width * 0.24)
        shift = int((1 - enter) * 65)
        left_x = pair_center_x - pair_offset - shift
        right_x = pair_center_x + pair_offset + shift
        center_y = int(self.height * 0.50)

        radius = int(min(self.width, self.height) * 0.225)
        thickness = max(10, int(radius * 0.22))
        graph_text_gap = int(self.height * 0.055)

        month_value = self.month_label.lower()
        sr, sg, sb = self.season_color
        top_segments = [
            ("Зібрано за ", self.font_label, (255, 255, 255, 240)),
            (month_value, self.font_label, (sr, sg, sb, 245)),
        ]
        _, top_h, _, _ = _measure_text_segments(draw, top_segments)
        top_center_y = int(center_y - radius - graph_text_gap - top_h / 2)
        draw_centered_segments(
            draw,
            top_segments,
            center_x=left_x,
            center_y=top_center_y,
            shadow_alpha=120,
        )

        current_ratio = self.ratio * chart_anim
        self._draw_donut(
            draw=draw,
            center=(left_x, center_y),
            radius=radius,
            thickness=thickness,
            progress_ratio=current_ratio,
            alpha=1.0,
        )

        percentage = current_ratio * 100
        percent_text = f"{percentage:.0f}%"
        draw_centered_text(
            draw,
            percent_text,
            center_x=left_x,
            center_y=center_y,
            font=self.font_percent,
            fill=(255, 255, 255, 245),
            shadow_alpha=110,
        )

        amount_segments = [
            (format_amount(self.settings.collected), self.font_number, (sr, sg, sb, 248)),
            (" із ", self.font_number, (238, 238, 238, 230)),
            (format_amount(self.settings.target), self.font_number, (238, 238, 238, 230)),
        ]
        _, amount_h, _, _ = _measure_text_segments(draw, amount_segments)
        amount_center_y = int(center_y + radius + graph_text_gap + amount_h / 2)
        draw_centered_segments(
            draw,
            amount_segments,
            center_x=left_x,
            center_y=amount_center_y,
            shadow_alpha=100,
        )

        qr_size = int(min(self.width, self.height) * 0.50)
        qr_text_gap = int(self.height * 0.022)
        donate_text = "Donate"
        donate_bbox = draw.textbbox((0, 0), donate_text, font=self.font_donate)
        donate_height = donate_bbox[3] - donate_bbox[1]
        donate_w = donate_bbox[2] - donate_bbox[0]
        donate_top_offset = donate_bbox[1]
        heart_char = "♥"
        heart_bbox = draw.textbbox((0, 0), heart_char, font=self.font_heart)
        heart_height = heart_bbox[3] - heart_bbox[1]
        heart_width = heart_bbox[2] - heart_bbox[0]
        heart_top_offset = heart_bbox[1]
        text_row_height = max(donate_height, heart_height)
        right_total_height = qr_size + qr_text_gap + text_row_height

        qr_y = center_y - right_total_height // 2
        qr_x = right_x - qr_size // 2

        qr_img = self.qr_image.resize((qr_size, qr_size), Image.Resampling.LANCZOS).copy()
        layer.alpha_composite(qr_img, (qr_x, qr_y))

        row_center_y = qr_y + qr_size + qr_text_gap + text_row_height // 2
        donate_y = row_center_y - donate_height // 2 - donate_top_offset
        heart_gap = int(self.height * 0.010)
        total_width = donate_w + heart_gap + heart_width
        text_x = right_x - total_width // 2

        shadow_color = (0, 0, 0, 95)
        draw.text((text_x + 2, donate_y + 2), donate_text, font=self.font_donate, fill=shadow_color)
        draw.text(
            (text_x, donate_y),
            donate_text,
            font=self.font_donate,
            fill=(255, 255, 255, 240),
        )
        heart_x = text_x + donate_w + heart_gap
        heart_y = row_center_y - heart_height // 2 - heart_top_offset
        draw.text((heart_x + 2, heart_y + 2), heart_char, font=self.font_heart, fill=(0, 0, 0, 85))
        draw.text((heart_x, heart_y), heart_char, font=self.font_heart, fill=(220, 18, 46, 255))

        overlay.alpha_composite(apply_layer_alpha(layer, section_alpha))

    def render_frame(self, background_rgb: np.ndarray, t_seconds: float, duration: float) -> np.ndarray:
        t_norm = clamp(t_seconds / max(duration, 1e-6))
        base = Image.fromarray(background_rgb.astype(np.uint8)).convert("RGBA")
        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        base_draw = ImageDraw.Draw(overlay)

        shade_alpha = int(88 + 30 * math.sin(t_norm * math.tau))
        base_draw.rectangle((0, 0, self.width, self.height), fill=(0, 0, 0, shade_alpha))

        self._draw_verse_section(overlay, t_norm=t_norm)
        self._draw_money_section(overlay, t_norm=t_norm)

        final = Image.alpha_composite(base, overlay).convert("RGB")
        return np.asarray(final)


def parse_args(args: Iterable[str] | None = None) -> RenderSettings:
    parser = argparse.ArgumentParser(description="Generate an animated tithing video over a background clip.")
    parser.add_argument("-t", "--target", required=True, type=float, help="Target amount for the month.")
    parser.add_argument("-c", "--collected", required=True, type=float, help="Collected amount so far.")
    parser.add_argument("-m", "--month", required=True, type=int, help="Month number in range 1..12.")
    parser.add_argument(
        "-b",
        "--background",
        required=True,
        type=Path,
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
        help="Path to static QR image. Placeholder is generated if missing.",
    )
    parser.add_argument("-f", "--fps", type=int, default=30, help="Output frames per second.")

    ns = parser.parse_args(args=args)
    if ns.target <= 0:
        parser.error("--target must be greater than 0.")
    if ns.collected < 0:
        parser.error("--collected must be non-negative.")
    if ns.month < 1 or ns.month > 12:
        parser.error("--month must be an integer in range 1..12.")
    if ns.fps <= 0:
        parser.error("--fps must be greater than 0.")
    if not ns.background.exists():
        parser.error(f"Background video does not exist: {ns.background}")
    if not ns.qr.exists():
        parser.error(f"QR file does not exist: {ns.qr}")

    return RenderSettings(
        target=ns.target,
        collected=ns.collected,
        month=ns.month,
        background_path=ns.background,
        output_path=ns.output,
        qr_path=ns.qr,
        fps=ns.fps,
    )


def render_video(settings: RenderSettings) -> None:
    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    settings.qr_path.parent.mkdir(parents=True, exist_ok=True)

    background_clip = VideoFileClip(str(settings.background_path))
    output_clip: VideoClip | None = None
    try:
        duration = float(background_clip.duration)
        width, height = background_clip.size
        renderer = SceneRenderer(width=width, height=height, settings=settings)

        def frame_fn(t: float) -> np.ndarray:
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

        output_clip.write_videofile(
            str(settings.output_path),
            fps=settings.fps,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset="medium",
        )
    finally:
        if output_clip is not None:
            output_clip.close()
        background_clip.close()


def main() -> None:
    settings = parse_args()
    render_video(settings)


if __name__ == "__main__":
    main()

