"""Rendering, captions and the assembled submission video.

Frames are rendered from stored joint positions/velocities with
``mujoco.Renderer`` (EGL on Colab GPUs) and streamed straight into MP4 writers,
so memory stays small even for a 1280x720, ~100 s video.

Text is drawn with Pillow. Japanese captions need a CJK font
(``apt-get install fonts-noto-cjk``); without one, pass English text.
"""

from __future__ import annotations

import functools
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import mediapy as media
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ACCENT = (255, 214, 102)
INK = (11, 11, 11)


# --------------------------------------------------------------------------- text
@functools.lru_cache(maxsize=None)
def font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size, index=0)  # index 0 = JP face
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def has_cjk_font() -> bool:
    return any(os.path.exists(p) for p in _FONT_CANDIDATES)


def fit_size(text: str, size: int, max_width: Optional[int]) -> int:
    """Largest font size <= ``size`` whose rendered text fits in ``max_width`` pixels."""
    if not max_width:
        return size
    while size > 10 and font(size).getlength(text) > max_width:
        size -= 2
    return size


def draw_text(
    img: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    size: int = 22,
    fill=WHITE,
    anchor: str = "la",
    stroke: int = 3,
    max_width: Optional[int] = None,
) -> np.ndarray:
    """Draw outlined text onto an RGB uint8 image (returns a new array).

    If ``max_width`` is given, the font shrinks until the text fits.
    """
    size = fit_size(text, size, max_width)
    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).text(
        xy, text, font=font(size), fill=fill, anchor=anchor,
        stroke_width=stroke, stroke_fill=BLACK,
    )
    return np.asarray(pil)


def caption_bar(img: np.ndarray, text: str, size: int = 26, height: int = 52) -> np.ndarray:
    """Darken the bottom band of a frame and write a caption in it."""
    out = img.copy()
    out[-height:] = (out[-height:] * 0.25).astype(np.uint8)
    return draw_text(out, text, (out.shape[1] // 2, out.shape[0] - height // 2), size=size,
                     anchor="mm", stroke=0, max_width=out.shape[1] - 40)


# ---------------------------------------------------------------------- rendering
class Renderer:
    """Re-usable offscreen renderer for one model and frame size."""

    def __init__(self, mj_model: mujoco.MjModel, height: int, width: int, camera: str = "track"):
        self.model = mj_model
        self.camera = camera
        self.renderer = mujoco.Renderer(mj_model, height=height, width=width)
        self.data = mujoco.MjData(mj_model)

    def render(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


def panel_overlay(
    img: np.ndarray,
    title: str,
    subtitle: Optional[str] = None,
    lines: Sequence[str] = (),
    scale: float = 1.0,
) -> np.ndarray:
    t_size, s_size = int(34 * scale), int(20 * scale)
    max_w = img.shape[1] - 32
    img = draw_text(img, title, (16, 12), size=t_size, fill=ACCENT, max_width=max_w)
    y = 20 + t_size
    if subtitle:
        img = draw_text(img, subtitle, (16, y), size=s_size, max_width=max_w)
        y += s_size + 10
    for line in lines:
        img = draw_text(img, line, (16, y), size=s_size)
        y += s_size + 6
    return img


def live_stats(speed: float, power_w: float, japanese: bool) -> List[str]:
    if japanese:
        return [f"速度 {speed:.2f} m/s", f"関節の消費パワー {power_w:4.0f} W"]
    return [f"speed {speed:.2f} m/s", f"joint power {power_w:4.0f} W"]


# ------------------------------------------------------------------------ writing
class MultiWriter:
    """Write the same frames to several MP4 files at once."""

    def __init__(self, paths: Iterable[str], height: int, width: int, fps: float):
        self.writers = [media.VideoWriter(str(p), shape=(height, width), fps=fps) for p in paths]
        self.height, self.width, self.fps = height, width, fps

    def __enter__(self):
        for w in self.writers:
            w.__enter__()
        return self

    def __exit__(self, *exc):
        for w in self.writers:
            w.__exit__(*exc)

    def add(self, frame: np.ndarray) -> None:
        for w in self.writers:
            w.add_image(frame)

    def hold(self, frame: np.ndarray, seconds: float) -> None:
        for _ in range(max(1, int(round(seconds * self.fps)))):
            self.add(frame)


class Tee:
    """Forward ``add(frame)`` to several writers."""

    def __init__(self, *writers):
        self.writers = writers

    def add(self, frame: np.ndarray) -> None:
        for w in self.writers:
            w.add(frame)


def text_card(lines: Sequence[str], height: int, width: int, sizes: Optional[Sequence[int]] = None) -> np.ndarray:
    """A black frame with centered text lines (first line highlighted)."""
    scale = width / 1280.0  # sizes are designed for 1280x720
    sizes = list(sizes or [52] + [32] * (len(lines) - 1))
    sizes = [fit_size(line, max(10, int(s * scale)), width - 80) for line, s in zip(lines, sizes)]
    img = np.zeros((height, width, 3), dtype=np.uint8)
    gap = int(22 * scale)
    y = (height - (sum(sizes) + gap * (len(lines) - 1))) // 2
    for i, (line, size) in enumerate(zip(lines, sizes)):
        img = draw_text(img, line, (width // 2, y + size // 2), size=size, anchor="mm",
                        fill=ACCENT if i == 0 else WHITE, stroke=0)
        y += size + gap
    return img


def image_card(image_path: str, height: int, width: int, caption: Optional[str] = None) -> np.ndarray:
    """A still image (e.g. a result figure) letter-boxed onto a white frame."""
    cap_h = 64 if caption else 0
    canvas = Image.new("RGB", (width, height), WHITE)
    fig = Image.open(image_path).convert("RGB")
    fig.thumbnail((width - 48, height - 48 - cap_h))
    canvas.paste(fig, ((width - fig.width) // 2, cap_h + (height - cap_h - fig.height) // 2))
    img = np.asarray(canvas)
    if caption:
        img = draw_text(img, caption, (width // 2, cap_h // 2 + 8), size=int(32 * width / 1280),
                        fill=INK, anchor="mm", stroke=0, max_width=width - 60)
    return img


def write_style_clip(
    writer: MultiWriter,
    renderer: Renderer,
    logs: dict,
    power_w: np.ndarray,
    title: str,
    subtitle: str,
    command_labels: Sequence[str],
    japanese: bool,
    start: int,
    stop: int,
    every: int = 2,
    seed_index: int = 0,
) -> None:
    """One style, full frame, with live stats and the current command."""
    speed = np.linalg.norm(logs["global_linvel"][seed_index, :, :2], axis=-1)
    stop = min(stop, logs["qpos"].shape[1])
    for t in range(start, stop, every):
        img = renderer.render(logs["qpos"][seed_index, t], logs["qvel"][seed_index, t])
        img = panel_overlay(img, title, subtitle, live_stats(speed[t], power_w[seed_index, t], japanese))
        img = caption_bar(img, ("指令: " if japanese else "command: ") + command_labels[t])
        writer.add(img)


def write_grid(
    writer: MultiWriter,
    renderer: Renderer,
    logs_list: Sequence[dict],
    power_list: Sequence[np.ndarray],
    titles: Sequence[str],
    command_labels: Sequence[str],
    japanese: bool,
    every: int = 2,
    ncols: int = 2,
    seed_index: int = 0,
    out_size: Optional[Tuple[int, int]] = None,
) -> None:
    """All styles side by side (same commands, same time).

    ``out_size`` (height, width) pads the tiled canvas to the writer's frame
    size; by default it is 2x2 panels.
    """
    h, w = renderer.renderer.height, renderer.renderer.width
    n = len(logs_list)
    nrows = int(np.ceil(n / ncols))
    out_h, out_w = out_size or (2 * h, 2 * w)
    steps = min(lg["qpos"].shape[1] for lg in logs_list)
    speeds = [np.linalg.norm(lg["global_linvel"][seed_index, :, :2], axis=-1) for lg in logs_list]
    for t in range(0, steps, every):
        canvas = np.zeros((nrows * h, ncols * w, 3), dtype=np.uint8)
        for k, lg in enumerate(logs_list):
            img = renderer.render(lg["qpos"][seed_index, t], lg["qvel"][seed_index, t])
            img = panel_overlay(img, titles[k], None, live_stats(speeds[k][t], power_list[k][seed_index, t], japanese), scale=0.8)
            r, c = divmod(k, ncols)
            canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
        frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        y0, x0 = (out_h - canvas.shape[0]) // 2, (out_w - canvas.shape[1]) // 2
        frame[y0 : y0 + canvas.shape[0], x0 : x0 + canvas.shape[1]] = canvas
        frame = caption_bar(frame, ("指令: " if japanese else "command: ") + command_labels[t])
        writer.add(frame)
