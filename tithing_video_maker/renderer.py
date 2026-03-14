from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .constants import MONTH_COLORS_HEX, VERSE_REF, VERSE_TEXT
from .graphics import (
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

        dummy_draw = ImageDraw.Draw(Image.new("RGBA", (4, 4), (0, 0, 0, 0)))
        max_text_width = int(self.width * 0.84)
        self.verse_wrapped = wrap_text(dummy_draw, VERSE_TEXT, self.font_verse, max_text_width)
        self.verse_spacing = max(10, int(self.height * 0.012))

        mr, mg, mb = self.month_color
        self.amount_segments = [
            (format_amount(self.settings.collected), self.font_number, (mr, mg, mb, 248)),
            (" із ", self.font_number, (238, 238, 238, 230)),
            (format_amount(self.settings.target), self.font_number, (238, 238, 238, 230)),
        ]
        _, self.amount_height, _, _ = measure_text_segments(dummy_draw, self.amount_segments)

        self.qr_size = int(min(self.width, self.height) * 0.53)
        self.qr_image_resized = self.qr_image.resize((self.qr_size, self.qr_size), Image.Resampling.LANCZOS)

    def _draw_verse_section(self, overlay: Image.Image, t_norm: float) -> None:
        start = 0.03
        end = 0.36
        alpha = segment_alpha(t_norm, start, end, fade_in_share=0.16, fade_out_share=0.11)
        if alpha <= 0:
            return

        layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        center_x = self.width // 2
        verse_y = int(self.height * 0.43)
        verse_size = draw_centered_multiline_text(
            draw,
            self.verse_wrapped,
            center_x=center_x,
            center_y=verse_y,
            font=self.font_verse,
            fill=(255, 255, 255, 255),
            spacing=self.verse_spacing,
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

        pair_center_x = self.width // 2
        pair_offset = int(self.width * 0.22)
        shift = int((1 - enter) * 65)
        left_x = pair_center_x - pair_offset - shift
        right_x = pair_center_x + pair_offset + shift
        center_y = int(self.height * 0.50)

        radius = int(min(self.width, self.height) * 0.225)
        thickness = max(10, int(radius * 0.22))
        graph_text_gap = int(self.height * 0.055)

        month_value = self.month_label_lower
        mr, mg, mb = self.month_color
        top_segments = [
            ("Зібрано за ", self.font_label, (255, 255, 255, 240)),
            (month_value, self.font_label, (mr, mg, mb, 245)),
        ]
        _, top_h, _, _ = measure_text_segments(draw, top_segments)
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

        amount_center_y = int(center_y + radius + graph_text_gap + self.amount_height / 2)
        draw_centered_segments(
            draw,
            self.amount_segments,
            center_x=left_x,
            center_y=amount_center_y,
            shadow_alpha=100,
        )

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
        right_total_height = self.qr_size + qr_text_gap + text_row_height

        qr_y = center_y - right_total_height // 2
        qr_x = right_x - self.qr_size // 2

        layer.alpha_composite(self.qr_image_resized, (qr_x, qr_y))

        row_center_y = qr_y + self.qr_size + qr_text_gap + text_row_height // 2
        donate_y = row_center_y - donate_height // 2 - donate_top_offset
        heart_gap = int(self.height * 0.010)
        total_width = donate_w + heart_gap + heart_width
        text_x = right_x - total_width // 2

        shadow_color = (0, 0, 0, 95)
        draw.text((text_x + 2, donate_y + 2), donate_text, font=self.font_donate, fill=shadow_color)
        draw.text((text_x, donate_y), donate_text, font=self.font_donate, fill=(255, 255, 255, 240))
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
