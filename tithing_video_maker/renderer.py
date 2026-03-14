from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .constants import MONTH_COLORS_HEX, VERSE_REF, VERSE_TEXT
from .graphics import (
    TextSegment,
    apply_layer_alpha,
    clamp,
    draw_centered_multiline_text,
    draw_centered_segments,
    draw_centered_text,
    ease_out_cubic,
    format_amount,
    hex_to_rgb,
    load_font,
    load_heart_font,
    measure_text_segments,
    month_label_ua,
    segment_alpha,
    segment_progress,
    wrap_text,
)
from .models import RenderSettings
from .qr import ensure_placeholder_qr


class SceneRenderer:
    _IMAGE_PADDING = 2

    def __init__(self, width: int, height: int, settings: RenderSettings):
        self.width = width
        self.height = height
        self.settings = settings
        self.month_label = month_label_ua(settings.month)
        self.month_label_lower = self.month_label.lower()
        self.month_color = hex_to_rgb(MONTH_COLORS_HEX[settings.month])
        self.ratio = 0.0 if settings.target <= 0 else settings.collected / settings.target
        self.qr_image = ensure_placeholder_qr(settings.qr_path)

        self.font_verse = load_font(max(36, int(height * 0.074)), bold=True)
        self.font_reference = load_font(max(30, int(height * 0.049)), bold=True)
        self.font_label = load_font(max(38, int(height * 0.070)), bold=True)
        self.font_number = load_font(max(34, int(height * 0.060)), bold=True)
        self.font_percent = load_font(max(50, int(height * 0.097)), bold=True)
        self.font_donate = load_font(max(34, int(height * 0.060)), bold=True)
        self.font_heart = load_heart_font(max(42, int(height * 0.074)))

        self._measure_draw = ImageDraw.Draw(Image.new("RGBA", (4, 4), (0, 0, 0, 0)))
        max_text_width = int(self.width * 0.84)
        self.verse_wrapped = wrap_text(self._measure_draw, VERSE_TEXT, self.font_verse, max_text_width)
        self.verse_spacing = max(10, int(self.height * 0.012))
        self.verse_gap = int(self.height * 0.13)
        verse_bbox = self._measure_draw.multiline_textbbox(
            (0, 0),
            self.verse_wrapped,
            font=self.font_verse,
            align="center",
            spacing=self.verse_spacing,
        )
        self.verse_text_height = int(math.ceil(verse_bbox[3] - verse_bbox[1]))
        self.verse_image = self._build_verse_image()
        self.verse_position = (
            self.width // 2 - self.verse_image.width // 2,
            int(self.height * 0.43) - self.verse_text_height // 2 - self._IMAGE_PADDING,
        )

        mr, mg, mb = self.month_color
        self.amount_segments: list[TextSegment] = [
            (format_amount(self.settings.collected), self.font_number, (mr, mg, mb, 248)),
            (" із ", self.font_number, (238, 238, 238, 230)),
            (format_amount(self.settings.target), self.font_number, (238, 238, 238, 230)),
        ]
        self.money_title_segments: list[TextSegment] = [
            ("Зібрано за ", self.font_label, (255, 255, 255, 240)),
            (self.month_label_lower, self.font_label, (mr, mg, mb, 245)),
        ]
        _, self.money_title_height, _, _ = measure_text_segments(self._measure_draw, self.money_title_segments)
        _, self.amount_height, _, _ = measure_text_segments(self._measure_draw, self.amount_segments)
        self.money_title_height = int(math.ceil(self.money_title_height))
        self.amount_height = int(math.ceil(self.amount_height))
        self.money_title_image = self._render_segments_image(self.money_title_segments, shadow_alpha=120)
        self.amount_row_image = self._render_segments_image(self.amount_segments, shadow_alpha=100)

        self.qr_size = int(min(self.width, self.height) * 0.53)
        self.qr_image_resized = self.qr_image.resize((self.qr_size, self.qr_size), Image.Resampling.LANCZOS)
        self.money_center_y = int(self.height * 0.50)
        self.pair_center_x = self.width // 2
        self.pair_offset = int(self.width * 0.22)
        self.radius = int(min(self.width, self.height) * 0.225)
        self.thickness = max(10, int(self.radius * 0.22))
        self.graph_text_gap = int(self.height * 0.055)
        self.qr_text_gap = int(self.height * 0.022)
        self.heart_gap = int(self.height * 0.010)
        self.donate_text = "Donate"
        self.heart_char = "\u2665"
        donate_bbox = self._measure_draw.textbbox((0, 0), self.donate_text, font=self.font_donate)
        heart_bbox = self._measure_draw.textbbox((0, 0), self.heart_char, font=self.font_heart)
        self.donate_height = int(math.ceil(donate_bbox[3] - donate_bbox[1]))
        self.donate_width = int(math.ceil(donate_bbox[2] - donate_bbox[0]))
        self.donate_top_offset = int(math.floor(donate_bbox[1]))
        self.heart_height = int(math.ceil(heart_bbox[3] - heart_bbox[1]))
        self.heart_width = int(math.ceil(heart_bbox[2] - heart_bbox[0]))
        self.heart_top_offset = int(math.floor(heart_bbox[1]))
        self.text_row_height = max(self.donate_height, self.heart_height)
        self.right_panel_image = self._build_right_panel_image()
        self.percent_image_cache: dict[str, Image.Image] = {}

    def _build_verse_image(self) -> Image.Image:
        verse_bbox = self._measure_draw.multiline_textbbox(
            (0, 0),
            self.verse_wrapped,
            font=self.font_verse,
            align="center",
            spacing=self.verse_spacing,
        )
        reference_bbox = self._measure_draw.multiline_textbbox((0, 0), VERSE_REF, font=self.font_reference, spacing=4)
        verse_width = int(math.ceil(verse_bbox[2] - verse_bbox[0]))
        verse_height = int(math.ceil(verse_bbox[3] - verse_bbox[1]))
        reference_width = int(math.ceil(reference_bbox[2] - reference_bbox[0]))
        reference_height = int(math.ceil(reference_bbox[3] - reference_bbox[1]))
        width = max(1, max(verse_width, reference_width) + self._IMAGE_PADDING * 2)
        height = max(1, verse_height + self.verse_gap + reference_height + self._IMAGE_PADDING * 2)
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        verse_center_y = self._IMAGE_PADDING + verse_height // 2
        reference_center_y = verse_center_y + verse_height // 2 + self.verse_gap + reference_height // 2
        draw_centered_multiline_text(
            draw,
            self.verse_wrapped,
            center_x=width // 2,
            center_y=verse_center_y,
            font=self.font_verse,
            fill=(255, 255, 255, 255),
            spacing=self.verse_spacing,
            shadow_alpha=120,
        )
        draw_centered_multiline_text(
            draw,
            VERSE_REF,
            center_x=width // 2,
            center_y=reference_center_y,
            font=self.font_reference,
            fill=(245, 245, 245, 240),
            spacing=4,
            shadow_alpha=110,
        )
        return image

    def _render_segments_image(self, segments: list[TextSegment], shadow_alpha: int) -> Image.Image:
        total_width, height, _, _ = measure_text_segments(self._measure_draw, segments)
        total_width = int(math.ceil(total_width))
        height = int(math.ceil(height))
        image = Image.new(
            "RGBA",
            (
                max(1, total_width + self._IMAGE_PADDING * 2),
                max(1, height + self._IMAGE_PADDING * 2),
            ),
            (0, 0, 0, 0),
        )
        draw = ImageDraw.Draw(image)
        draw_centered_segments(
            draw,
            segments,
            center_x=image.width // 2,
            center_y=image.height // 2,
            shadow_alpha=shadow_alpha,
        )
        return image

    def _render_text_image(
        self,
        text: str,
        *,
        font: object,
        fill: tuple[int, int, int, int],
        shadow_alpha: int,
    ) -> Image.Image:
        bbox = self._measure_draw.textbbox((0, 0), text, font=font)
        width = int(math.ceil(bbox[2] - bbox[0]))
        height = int(math.ceil(bbox[3] - bbox[1]))
        image = Image.new(
            "RGBA",
            (
                max(1, width + self._IMAGE_PADDING * 2),
                max(1, height + self._IMAGE_PADDING * 2),
            ),
            (0, 0, 0, 0),
        )
        draw = ImageDraw.Draw(image)
        draw_centered_text(
            draw,
            text,
            center_x=image.width // 2,
            center_y=image.height // 2,
            font=font,
            fill=fill,
            shadow_alpha=shadow_alpha,
        )
        return image

    def _build_right_panel_image(self) -> Image.Image:
        total_width = self.donate_width + self.heart_gap + self.heart_width
        width = max(1, max(self.qr_size, total_width) + self._IMAGE_PADDING * 2)
        height = max(1, self.qr_size + self.qr_text_gap + self.text_row_height + self._IMAGE_PADDING * 2)
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        qr_x = width // 2 - self.qr_size // 2
        qr_y = self._IMAGE_PADDING
        image.alpha_composite(self.qr_image_resized, (qr_x, qr_y))

        row_center_y = qr_y + self.qr_size + self.qr_text_gap + self.text_row_height // 2
        donate_y = row_center_y - self.donate_height // 2 - self.donate_top_offset
        text_x = width // 2 - total_width // 2
        shadow_color = (0, 0, 0, 95)
        draw.text((text_x + 2, donate_y + 2), self.donate_text, font=self.font_donate, fill=shadow_color)
        draw.text((text_x, donate_y), self.donate_text, font=self.font_donate, fill=(255, 255, 255, 240))
        heart_x = text_x + self.donate_width + self.heart_gap
        heart_y = row_center_y - self.heart_height // 2 - self.heart_top_offset
        draw.text((heart_x + 2, heart_y + 2), self.heart_char, font=self.font_heart, fill=(0, 0, 0, 85))
        draw.text((heart_x, heart_y), self.heart_char, font=self.font_heart, fill=(220, 18, 46, 255))
        return image

    def _get_percent_image(self, percent_text: str) -> Image.Image:
        cached = self.percent_image_cache.get(percent_text)
        if cached is None:
            cached = self._render_text_image(
                percent_text,
                font=self.font_percent,
                fill=(255, 255, 255, 245),
                shadow_alpha=110,
            )
            self.percent_image_cache[percent_text] = cached
        return cached

    @staticmethod
    def _composite_with_alpha(
        overlay: Image.Image,
        image: Image.Image,
        position: tuple[int, int],
        alpha: float,
    ) -> None:
        if alpha <= 0:
            return
        layer = image if alpha >= 0.999 else apply_layer_alpha(image.copy(), alpha)
        overlay.alpha_composite(layer, position)

    def _draw_verse_section(self, overlay: Image.Image, t_norm: float) -> None:
        start = 0.03
        end = 0.36
        alpha = segment_alpha(t_norm, start, end, fade_in_share=0.16, fade_out_share=0.11)
        if alpha <= 0:
            return

        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        self._composite_with_alpha(layer, self.verse_image, self.verse_position, 1.0)
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
            mr, mg, mb = self.month_color
            draw.arc(
                (x0, y0, x1, y1),
                start=-90,
                end=-90 + 360 * main_ratio,
                fill=(mr, mg, mb, int(255 * alpha)),
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
        section_alpha = segment_alpha(t_norm, start, end, fade_in_share=0.09, fade_out_share=0.10)
        if section_alpha <= 0:
            return

        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        enter = ease_out_cubic(segment_progress(t_norm, start, start + 0.06))
        chart_anim = ease_out_cubic(segment_progress(t_norm, start + 0.02, start + 0.09))
        shift = int((1 - enter) * 65)
        left_x = self.pair_center_x - self.pair_offset - shift
        right_x = self.pair_center_x + self.pair_offset + shift

        top_center_y = int(self.money_center_y - self.radius - self.graph_text_gap - self.money_title_height / 2)
        self._composite_with_alpha(
            layer,
            self.money_title_image,
            (left_x - self.money_title_image.width // 2, top_center_y - self.money_title_image.height // 2),
            1.0,
        )

        current_ratio = self.ratio * chart_anim
        self._draw_donut(
            draw=draw,
            center=(left_x, self.money_center_y),
            radius=self.radius,
            thickness=self.thickness,
            progress_ratio=current_ratio,
            alpha=1.0,
        )

        percent_image = self._get_percent_image(f"{current_ratio * 100:.0f}%")
        self._composite_with_alpha(
            layer,
            percent_image,
            (left_x - percent_image.width // 2, self.money_center_y - percent_image.height // 2),
            1.0,
        )

        amount_center_y = int(self.money_center_y + self.radius + self.graph_text_gap + self.amount_height / 2)
        self._composite_with_alpha(
            layer,
            self.amount_row_image,
            (left_x - self.amount_row_image.width // 2, amount_center_y - self.amount_row_image.height // 2),
            1.0,
        )

        self._composite_with_alpha(
            layer,
            self.right_panel_image,
            (right_x - self.right_panel_image.width // 2, self.money_center_y - self.right_panel_image.height // 2),
            1.0,
        )

        overlay.alpha_composite(apply_layer_alpha(layer, section_alpha))

    def render_frame(self, background_rgb: np.ndarray, t_seconds: float, duration: float) -> np.ndarray:
        t_norm = clamp(t_seconds / max(duration, 1e-6))
        if background_rgb.dtype != np.uint8:
            background_rgb = background_rgb.astype(np.uint8)
        base = Image.fromarray(background_rgb).convert("RGBA")
        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        base_draw = ImageDraw.Draw(overlay)

        shade_alpha = int(88 + 30 * math.sin(t_norm * math.tau))
        base_draw.rectangle((0, 0, self.width, self.height), fill=(0, 0, 0, shade_alpha))

        self._draw_verse_section(overlay, t_norm=t_norm)
        self._draw_money_section(overlay, t_norm=t_norm)

        final = Image.alpha_composite(base, overlay).convert("RGB")
        return np.asarray(final)
