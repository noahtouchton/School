import json
import time
import random
from pynput import mouse, keyboard
from nordvpn_connect import initialize_vpn, rotate_VPN, close_vpn_connection

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
URL_TO_TYPE = "https://www.jacksonville.com/story/sports/high-school/soccer/2025/11/18/fhsaa-high-school-girls-soccer-jacksonville-player-fan-poll/87317134007/"
INPUT_FILE = "input_log.json"
LOOPS = 10000   # <--- SET HOW MANY TIMES YOU WANT IT TO RUN

# Cities to choose from each loop
CITIES = [
    "new york",
    "los angeles",
    "chicago",
    "dallas",
    "denver",
    "houston",
    "seattle",
    "phoenix",
    "san francisco",
    "buffalo",
    "kansas city",
    "nashville",
    "atlanta",
    "charlotte",
    "omaha",
    "pittsburgh",
    
]


# -------------------------------------------------
# Load events
# -------------------------------------------------
with open(INPUT_FILE, "r") as f:
    events = json.load(f)

events.sort(key=lambda e: e.get("time", 0.0))

mouse_controller = mouse.Controller()
keyboard_controller = keyboard.Controller()

stop_flag = False
vpn_settings = None  # will hold current VPN settings, if any

# -------------------------------------------------
# STOP IF USER PRESSES 'y'
# -------------------------------------------------
def on_key_press(key):
    global stop_flag
    try:
        if key.char == '`':  # pressing y stops script
            print("\n[STOP] 'y' key detected. Ending playback...")
            stop_flag = True
    except:
        pass

listener = keyboard.Listener(on_press=on_key_press)
listener.start()


def send_special_key(key_str: str):
    try:
        name = key_str.split(".", 1)[1]  # 'Key.enter' -> 'enter'
        key_obj = getattr(keyboard.Key, name)
        keyboard_controller.press(key_obj)
        keyboard_controller.release(key_obj)
    except Exception as e:
        print(f"[WARN] Could not send special key {key_str}: {e}")


# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------
print("Starting playback... Press 'y' to stop.")

for i in range(LOOPS):
    if stop_flag:
        break

    print(f"\n--- Loop {i+1}/{LOOPS} ---")

    for ev in events:
        if stop_flag:
            break

        dt = float(ev.get("dt", 0.0))
        dt = min(dt, 2.0)
        if dt > 0:
            time.sleep(dt)

        ev_type = ev.get("type")

        if ev_type == "mouse_click":
            button_str = ev.get("button", "Button.left")
            x, y = ev.get("position", [0, 0])

            if "right" in button_str:
                button_obj = mouse.Button.right
            elif "middle" in button_str:
                button_obj = mouse.Button.middle
            else:
                button_obj = mouse.Button.left

            mouse_controller.position = (x, y)
            mouse_controller.click(button_obj, 1)
            print(f"[PLAY] Click {button_str} at {(x, y)}")

        elif ev_type == "key_press":
            key_str = ev.get("key", "")

            if key_str.startswith("Key."):
                print(f"[PLAY] Special key {key_str}")
                send_special_key(key_str)
            else:
                print(f"[PLAY] Char key '{key_str}'")
                keyboard_controller.type(key_str)

        elif ev_type == "url_marker":
            print(f"[PLAY] URL marker -> typing URL: {URL_TO_TYPE}")
            keyboard_controller.type(URL_TO_TYPE)

        else:
            print(f"[WARN] Unknown event type: {ev_type}")

    if stop_flag:
        break

    # -------------------------------------------------
    # SWITCH VPN TO RANDOM CITY
    # -------------------------------------------------
    city = random.choice(CITIES)
    print(f"[VPN] Switching VPN location to: {city}")
    vpn_settings = initialize_vpn(city)  # starts nordvpn and stuff for this city
    rotate_VPN(vpn_settings)             # actually connect to server
    time.sleep(1)

print("Playback finished.")

# Close VPN connection if we ever opened one
if vpn_settings is not None:
    close_vpn_connection(vpn_settings)
