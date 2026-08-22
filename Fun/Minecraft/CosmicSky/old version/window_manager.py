import pygetwindow as gw

import pygetwindow as gw

import ctypes
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

def get_minecraft_window():
    all_windows = gw.getWindowsWithTitle("Minecraft")
    for win in all_windows:
        title = win.title.lower()
        if "launcher" in title or "pycharm" in title:
            continue
        if "minecraft" in title and ("multiplayer" in title or "singleplayer" in title or "server" in title):
            print(f"✅ Using game window: {win.title}")
            return (win.left, win.top, win.width, win.height)
    raise RuntimeError("Minecraft game window not found")


def get_absolute_pos(relative_pos, win_offset):
    return (win_offset[0] + relative_pos[0], win_offset[1] + relative_pos[1])

def get_absolute_rect(relative_rect, win_offset):
    x1, y1, x2, y2 = relative_rect
    abs_x1, abs_y1 = get_absolute_pos((x1, y1), win_offset)
    abs_x2, abs_y2 = get_absolute_pos((x2, y2), win_offset)
    return (abs_x1, abs_y1, abs_x2, abs_y2)

def convert_absolute_to_relative_config(absolute_config, win_offset):
    relative_config = {}
    for key, value in absolute_config.items():
        if isinstance(value, tuple) and len(value) == 2:
            #single pixel
            relative_config[key] = (
                value[0] - win_offset[0],
                value[1] - win_offset[1]
            )
        elif isinstance(value, tuple) and len(value) == 4:
            #rectangle
            relative_config[key] = (
                value[0] - win_offset[0],
                value[1] - win_offset[1],
                value[2] - win_offset[0],
                value[3] - win_offset[1]
            )
        else:
            raise ValueError(f"Unsupported config value for ket '{key}': {value}")
    return relative_config


from PIL import Image, ImageDraw
import pyautogui

def visualize_config_positions(config, win_offset, output_path="config_preview.png", circle_radius=10):
    """
    Takes a screenshot and highlights all single-point config positions with red circles.
    Also outlines crop regions in green.
    """
    img = pyautogui.screenshot().convert("RGB")
    draw = ImageDraw.Draw(img)

    # Highlight single pixel positions
    for key, value in config.items():
        if isinstance(value, tuple) and len(value) == 2:
            abs_x, abs_y = get_absolute_pos(value, win_offset)
            draw.ellipse(
                [abs_x - circle_radius, abs_y - circle_radius,
                 abs_x + circle_radius, abs_y + circle_radius],
                outline="red", width=3
            )
            draw.text((abs_x + 5, abs_y + 5), key, fill="red")

        elif isinstance(value, tuple) and len(value) == 4:
            abs_x1, abs_y1, abs_x2, abs_y2 = get_absolute_rect(value, win_offset)
            draw.rectangle([abs_x1, abs_y1, abs_x2, abs_y2], outline="green", width=3)
            draw.text((abs_x1 + 5, abs_y1 + 5), key, fill="green")

    img.save(output_path)
    print(f"📸 Config visualization saved to: {output_path}")

import pyautogui
import keyboard
import time

def collect_points(keys_to_labels):
    print("🖱️ Hover over each point and press the matching key:")
    print("Press [Enter] when done.\n")
    points = {}
    while True:
        for key, label in keys_to_labels.items():
            if keyboard.is_pressed(key):
                x, y = pyautogui.position()
                points[label] = (x, y)
                print(f"{label}: ({x}, {y})")
                time.sleep(0.5)
        if keyboard.is_pressed("enter"):
            print("✅ Finished collecting points.")
            return points

keys = {
    "c": "captcha_check",
    "a": "alignment_check",
    "s": "spawn_check",
    "f": "fishing_pole_check",
    "1": "fishout_check_top_left",
    "2": "fishout_check_bottom_right",
    "x": "image_crop_top_left",
    "z": "image_crop_bottom_right"
}

monitor_configs = {
    "default_absolute": {
        "captcha_check": (1341, 732),
        "alignment_check": (846, 884),
        "spawn_check": (1341, 1369),
        "fishing_pole_check": (1111, 1381),
        "fishout_check_top_left": (1015, 1179),
        "fishout_check_bottom_right": (1020, 1180),
        "image_crop": (1256, 660, 1300, 740)
    },

    "base_absolute": {
        "spawn_check": (1019, 1008),
        "alignment_check": (684, 647),
        "captcha_check": (1023, 588),
        "fishing_pole_check": (790, 1021),

        "fishout_check_rect": (
            735, 840, 768, 841
        ),

        "image_crop": (
            940, 482, 1000, 580
        )
    }
}

