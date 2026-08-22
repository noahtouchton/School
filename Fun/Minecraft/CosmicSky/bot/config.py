# config.py
import os

# --- Window Configuration ---
WINDOW_TITLES = ["Minecraft", "LabyMod", "Lunar Client", "Badlion"]
DPI_AWARE = True  # Enable SetProcessDPIAware

# --- Safety Controls ---
EXIT_KEY = "y"  # Press this key to emergency exit the bot
PAUSE_KEY = "p"  # Toggle pause

# --- Slaying Configuration ---
# Click interval ranges in seconds (simulating human jitter)
SLAY_CLICK_MIN_DELAY = 0.05
SLAY_CLICK_MAX_DELAY = 0.15
SLAY_CLICK_COOLDOWN = 0.2  # Pause between target switches
# Angles/Directions targets can be calibrated (yaw, pitch) relative or absolute
SLAY_TARGETS = [
    {"yaw": 0.0, "pitch": 0.0},
    {"yaw": 90.0, "pitch": 10.0},
    {"yaw": 180.0, "pitch": 0.0},
    {"yaw": 270.0, "pitch": 10.0},
]

# --- Mining Configuration ---
# Predetermined path represented as movement actions
# Each node: duration of holding W, rotation target (relative yaw change or absolute), block mining time
MINING_PATH = [
    {"yaw": 0, "pitch": 55, "move_duration": 0.5, "mine_duration": 2.0},
    {"yaw": 15, "pitch": 55, "move_duration": 0.5, "mine_duration": 2.0},
    {"yaw": 30, "pitch": 55, "move_duration": 0.5, "mine_duration": 2.0},
    {"yaw": 45, "pitch": 55, "move_duration": 0.5, "mine_duration": 2.0},
    # Moves in a circular pattern breaking blocks in front/down
]

# --- Fishing Configuration ---
FISHING_POLE_SLOT = "3"
WEAPON_SLOT = "2"
anti_idle_interval = 14 * 60 + 20  # 14m40s

# Relative crop regions for ML fishing (percentages of window size: x1_rel, y1_rel, x2_rel, y2_rel)
FISHING_CROP_REL = (0.45, 0.45, 0.55, 0.55)  # Center of screen for particles
FISH_DETECTOR_MODEL_PATH = "fish_detector.pth"

# --- Sell Configuration ---
SELL_CHAT_COMMAND = "/sell all"
SELL_INTERVAL_SEC = 300  # Sell every 5 minutes

# --- Gemini Captcha Solver ---
# Prompt sent to Gemini API with the screenshot
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_PROMPT = """
This is a screenshot of a Minecraft inventory-style puzzle/captcha.
Look at the text/item displayed at the top of the container GUI.
It asks you to click a specific target item in the grid below it.
Find the slot where the requested item is located in the chest grid.
Identify:
1. Is a puzzle GUI currently open on the screen? (is_captcha)
2. What is the target item name? (target_item)
3. What are the (X, Y) pixel coordinates of the exact center of that item in the provided screenshot? (click_coords)

Return ONLY a JSON response in the following format:
{
  "is_captcha": true/false,
  "target_item": "item_name_or_none",
  "click_coords": {"x": X_PIXEL, "y": Y_PIXEL}
}
"""
# If Gemini API Key is not set in env, look at local config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

# Pushover notification credentials (set these in your environment)
PUSHOVER_USER_KEY = os.environ.get("PUSHOVER_USER_KEY", "")
PUSHOVER_API_TOKEN = os.environ.get("PUSHOVER_API_TOKEN", "")
