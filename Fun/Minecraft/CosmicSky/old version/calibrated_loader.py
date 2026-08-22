# calibrated_loader.py
import json
from window_manager import get_minecraft_window

def load_calibrated(path="calibrated_config.json"):
    with open(path, "r", encoding="utf-8") as f:
        C = json.load(f)
    return C

def _rel_to_abs_point(xr, yr, win):
    wx, wy, ww, wh = win
    return (int(wx + xr * ww), int(wy + yr * wh))

def _rel_to_abs_rect(x1r, y1r, x2r, y2r, win):
    wx, wy, ww, wh = win
    return (
        int(wx + x1r * ww), int(wy + y1r * wh),
        int(wx + x2r * ww), int(wy + y2r * wh)
    )

def realize_config(C, win=None):
    """
    Returns absolute (screen) coords for current Minecraft window,
    plus color + margins dictionaries you can use directly.
    """
    if win is None:
        win = get_minecraft_window()

    points_abs = {}
    colors = {}
    margins = {"default": C.get("margins", {}).get("default", 8)}

    for name, data in C["points"].items():
        xr, yr = data["xy_rel"]
        points_abs[name] = _rel_to_abs_point(xr, yr, win)
        colors[name] = tuple(data["rgb"])  # (R,G,B)

    rects_abs = {}
    for name, data in C["rects"].items():
        x1r, y1r, x2r, y2r = data["rect_rel"]
        rects_abs[name] = _rel_to_abs_rect(x1r, y1r, x2r, y2r, win)

    return points_abs, rects_abs, colors, margins, win
