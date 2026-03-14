from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .graphics import load_font


def create_placeholder_qr_image(size: int = 1024) -> Image.Image:
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


def svg_number(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    text = value.strip()
    match = re.match(r"[-+]?\d*\.?\d+", text)
    if not match:
        return default
    return float(match.group(0))


def parse_rotate_center(transform: str | None) -> tuple[float, float] | None:
    if not transform:
        return None
    match = re.search(
        r"rotate\(\s*[-+]?\d*\.?\d+\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)",
        transform,
    )
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def looks_like_black_square(image: Image.Image) -> bool:
    arr = np.asarray(image.convert("RGBA"))
    alpha = arr[..., 3] > 0
    if alpha.sum() == 0:
        return False
    rgb = arr[..., :3]
    black = ((rgb == 0).all(axis=-1) & alpha).sum()
    ratio = float(black) / float(alpha.sum())
    unique_colors = len(np.unique(arr.reshape(-1, 4), axis=0))
    return ratio > 0.98 and unique_colors <= 2


def paint_finder_pattern(matrix: np.ndarray, top: int, left: int) -> None:
    n = matrix.shape[0]
    for y in range(7):
        for x in range(7):
            gy = top + y
            gx = left + x
            if gy < 0 or gx < 0 or gy >= n or gx >= n:
                continue
            is_black = x in (0, 6) or y in (0, 6) or (2 <= x <= 4 and 2 <= y <= 4)
            matrix[gy, gx] = is_black


def render_stylized_qr_svg_as_grid(path: Path, size: int = 1024) -> Image.Image:
    tree = ET.parse(path)
    root = tree.getroot()

    width = max(1, int(round(svg_number(root.attrib.get("width"), 500.0))))
    height = max(1, int(round(svg_number(root.attrib.get("height"), 500.0))))

    dot_clip = root.find(".//{*}clipPath[@id='clip-path-dot-color']")
    if dot_clip is None:
        raise RuntimeError("Unsupported SVG QR format: clip-path-dot-color was not found.")

    module_centers: list[tuple[float, float]] = []
    module_sizes: list[float] = []

    for elem in dot_clip:
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "rect":
            x = svg_number(elem.attrib.get("x"))
            y = svg_number(elem.attrib.get("y"))
            w = svg_number(elem.attrib.get("width"))
            h = svg_number(elem.attrib.get("height"))
            if w > 0 and h > 0:
                module_centers.append((x + w / 2.0, y + h / 2.0))
                module_sizes.extend([w, h])
            continue

        if tag == "circle":
            cx = svg_number(elem.attrib.get("cx"))
            cy = svg_number(elem.attrib.get("cy"))
            r = svg_number(elem.attrib.get("r"))
            module_centers.append((cx, cy))
            if r > 0:
                module_sizes.append(r * 2.0)
            continue

        if tag == "path":
            center = parse_rotate_center(elem.attrib.get("transform"))
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

    paint_finder_pattern(matrix, top=0, left=0)
    paint_finder_pattern(matrix, top=0, left=grid_size - 7)
    paint_finder_pattern(matrix, top=grid_size - 7, left=0)

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


def load_svg_qr(path: Path, size: int = 1024) -> Image.Image:
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

    if image is None or looks_like_black_square(image):
        image = render_stylized_qr_svg_as_grid(path, size=size)

    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def ensure_placeholder_qr(path: Path, size: int = 1024) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"QR file does not exist: {path}")
    if path.suffix.lower() == ".svg":
        return load_svg_qr(path, size=size)
    return Image.open(path).convert("RGBA")
