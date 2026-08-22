#!/usr/bin/env python3
import os
import time
from datetime import datetime
import requests
import keyboard
import pyautogui
from PIL import Image  # noqa: F401  (used by pyautogui.screenshot().convert)
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18

from window_manager import get_minecraft_window, visualize_config_positions
from calibrated_loader import load_calibrated, realize_config

# ---------- Pushover (notify) ----------
PUSHOVER_USER_KEY = os.environ.get("PUSHOVER_USER_KEY", "")
PUSHOVER_API_TOKEN = os.environ.get("PUSHOVER_API_TOKEN", "")

def send_notification(text):
    try:
        data = {
            "token": PUSHOVER_API_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "title": "Minecraft Bot",
            "message": text,
            "priority": 1,
        }
        r = requests.post("https://api.pushover.net/1/messages.json", data=data, timeout=5)
        if r.status_code == 200:
            print("Pushover notification sent.")
        else:
            print(f"Failed to send Pushover notification: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"Pushover error: {e}")

# ---------- Paths ----------
SAVE_DIR = "fishing_data"
POSITIVE_DIR = os.path.join(SAVE_DIR, "fish")
NEGATIVE_DIR = os.path.join(SAVE_DIR, "nofish")
os.makedirs(POSITIVE_DIR, exist_ok=True)
os.makedirs(NEGATIVE_DIR, exist_ok=True)

# ---------- Config + Helpers ----------
def rgb_close(px, tgt, margin):
    return all(abs(int(px[i]) - int(tgt[i])) <= int(margin) for i in range(3))

def sample_pixel(pos):
    # pos is absolute (x,y). Single screenshot per call site is fine here.
    return pyautogui.screenshot().getpixel(pos)

def sample_region(rect):
    x1, y1, x2, y2 = rect
    w, h = x2 - x1, y2 - y1
    return pyautogui.screenshot(region=(x1, y1, w, h)).convert("RGB")

def countdown_secs(s):
    for i in range(s, 0, -1):
        print(f"{i}...", end="\r", flush=True)
        time.sleep(1)
    print("   ", end="\r", flush=True)

# ---------- Model ----------
def load_model():
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("fish_detector.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# ImageNet normalization (change only if your training used 0.5/0.5):
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------- Bot Actions ----------
def fisherman_fred():
    pyautogui.press("3")
    pyautogui.rightClick()
    pyautogui.press("2")

def ditch_hotbar():
    for key in ("4", "5", "6"):
        pyautogui.press(key)
        pyautogui.press("q")
        time.sleep(0.1)
    pyautogui.press("x")
    pyautogui.press("q")
    time.sleep(0.1)
    pyautogui.press("2")

def save_labeled_image(label, crop_rect_abs):
    x1, y1, x2, y2 = crop_rect_abs
    w, h = x2 - x1, y2 - y1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img = pyautogui.screenshot(region=(x1, y1, w, h)).convert("RGB")
    fn = f"{ts}.png"
    if label == "fish":
        img.save(os.path.join(POSITIVE_DIR, fn))
        print(f"Saved fish image: {fn}")
    elif label == "nofish":
        img.save(os.path.join(NEGATIVE_DIR, fn))
        print(f"Saved nofish image: {fn}")

from PIL import ImageDraw, ImageFont  # at top with PIL import

def make_regions_overlay(points_abs, rects_abs, win_rect, filename="regions_overlay.png"):
    """
    Saves a window-cropped PNG with boxes/crosshairs + labels for all regions.
    - Rects: thick boxes (blue for image_crop, green for fishout)
    - Points: crosshairs + small circle (red)
    """
    # Fullscreen screenshot
    full = pyautogui.screenshot().convert("RGB")
    wx, wy, ww, wh = win_rect

    # Crop to the Minecraft window for precision view
    win_img = full.crop((wx, wy, wx + ww, wy + wh))
    draw = ImageDraw.Draw(win_img)

    def shift(p):  # shift absolute -> window-local
        return (p[0] - wx, p[1] - wy)

    def draw_point(name, xy, color="red"):
        x, y = shift(xy)
        r = 8
        # crosshair
        draw.line((x - 14, y, x + 14, y), fill=color, width=2)
        draw.line((x, y - 14, x, y + 14), fill=color, width=2)
        # small circle
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
        # label
        draw.text((x + 10, y + 10), name, fill=color)

    def draw_rect(name, rect, color="green"):
        x1, y1, x2, y2 = rect
        x1, y1 = shift((x1, y1))
        x2, y2 = shift((x2, y2))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        # label at top-left
        draw.text((x1 + 6, y1 + 6), name, fill=color)

    # Draw rects
    if "image_crop" in rects_abs:
        draw_rect("image_crop", rects_abs["image_crop"], color="blue")
    if "fishout_check_rect" in rects_abs:
        draw_rect("fishout_check_rect", rects_abs["fishout_check_rect"], color="green")

    # Draw points
    for key in ("captcha_check", "alignment_check", "spawn_check", "fishing_pole_check"):
        if key in points_abs:
            draw_point(key, points_abs[key], color="red")

    win_img.save(filename)
    print(f"📸 Regions overlay saved → {filename}")

# ---------- Checks (all driven by calibrated JSON) ----------
class Checks:
    def __init__(self, points, rects, colors, margins):
        self.P = points
        self.R = rects
        self.C = colors
        self.M = margins or {}
        self.M.setdefault("default", 8)

    def margin(self, key):
        return self.M.get(key, self.M["default"])

    def is_captcha(self):
        pos = self.P["captcha_check"]
        target = self.C["captcha_check"]
        return rgb_close(sample_pixel(pos), target, self.margin("captcha_check"))

    def is_aligned(self):
        # Your original logic: MATCH -> not aligned. We invert to return "aligned".
        pos = self.P["alignment_check"]
        target = self.C["alignment_check"]
        not_aligned = rgb_close(sample_pixel(pos), target, self.margin("alignment_check"))
        return not_aligned

    def in_spawn(self):
        pos = self.P["spawn_check"]
        target = self.C["spawn_check"]
        return rgb_close(sample_pixel(pos), target, self.margin("spawn_check"))

    def fishout(self):
        # Scan red-ish in rect; you can add a specific color to JSON if needed.
        x1, y1, x2, y2 = self.R["fishout_check_rect"]
        img = sample_region((x1, y1, x2, y2))
        w, h = x2 - x1, y2 - y1
        target = (252, 84, 84)
        m = max(12, self.margin("fishout_check_rect"))
        for xx in range(w):
            for yy in range(h):
                if rgb_close(img.getpixel((xx, yy)), target, m):
                    return True
        return False

    def ml_predict_fish(self, model):
        x1, y1, x2, y2 = self.R["image_crop"]
        w, h = x2 - x1, y2 - y1
        img = pyautogui.screenshot(region=(x1, y1, w, h)).convert("RGB")
        t = TRANSFORM(img).unsqueeze(0)
        with torch.no_grad():
            out = model(t)
            pred = torch.argmax(out, dim=1).item()
        return pred == 1

# ---------- Main loop ----------
def fisher():
    # Load & realize calibration
    CAL = load_calibrated("calibrated_config.json")
    points_abs, rects_abs, colors, margins, win = realize_config(CAL)
    print(f"🪟 Minecraft window: x={win[0]} y={win[1]} w={win[2]} h={win[3]}")

    # Optional overlay preview so you can sanity-check positions:
    try:
        preview = {}
        preview.update(points_abs)
        preview.update(rects_abs)
        visualize_config_positions(preview, (0, 0, 0, 0), output_path="config_preview.png")
        print("📸 Config visualization saved to: config_preview.png")
    except Exception as e:
        print(f"(Preview skipped: {e})")

    Ck = Checks(points_abs, rects_abs, colors, margins)

    # Prep runtime
    print("Starting in 5s...")
    countdown_secs(5)
    #ditch_hotbar()
    model = load_model()
    last_fisherman_time = time.time()
    fisherman_fred()

    counter = 0
    noti_flag = True
    was_misaligned = False

    print("Press 'y' to stop.")
    while True:
        counter += 1

        # Every 10 ticks: spawn check
        if counter % 10 == 0 and Ck.in_spawn():
            print("In spawn — exiting.")
            break

        # Captcha gate
        if Ck.is_captcha():
            print("Please solve captcha.")
            if noti_flag:
                send_notification("Captcha Detected")
                noti_flag = False
            time.sleep(5)
            continue
        else:
            noti_flag = True

        # Alignment gate
        if not Ck.is_aligned():
            if not was_misaligned:
                print("Not aligned")
                was_misaligned = True
            time.sleep(1)
            continue
        else:
            if was_misaligned:
                print("Aligned")
                was_misaligned = False

        # Periodic fish-out scan
        if counter % 25 == 0 and Ck.fishout():
            send_notification("Fished out")
            break

        # ML detection
        if Ck.ml_predict_fish(model):
            print("Fish detected (ML)")
            pyautogui.rightClick()
            time.sleep(3)

        # Labeling
        if keyboard.is_pressed("g"):
            #save_labeled_image("fish", rects_abs["image_crop"])
            time.sleep(0.5)
        elif keyboard.is_pressed("h"):
            #save_labeled_image("nofish", rects_abs["image_crop"])
            time.sleep(0.5)

        # Safety recast every 30 ticks based on fishing_pole_check color
        if counter % 30 == 0:
            pos = points_abs["fishing_pole_check"]
            target = colors["fishing_pole_check"]
            if rgb_close(sample_pixel(pos), target, Ck.margin("fishing_pole_check")):
                pyautogui.rightClick()
                time.sleep(1)

        # Manual quit
        if keyboard.is_pressed("y"):
            print("Key 'y' pressed — exiting.")
            break

        # Periodic hotbar clean
        if counter % 1000 == 0:
            #ditch_hotbar()
            continue

        # Periodic recast / anti-idle (~17m40s)
        if time.time() - last_fisherman_time >= 14 * 60 + 20:
            print("Calling fisherman_fred…")
            fisherman_fred()
            last_fisherman_time = time.time()
            #ditch_hotbar()

def main():
    fisher()

if __name__ == "__main__":
    main()
