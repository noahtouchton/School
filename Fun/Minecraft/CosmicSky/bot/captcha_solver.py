# captcha_solver.py
import base64
import json
import time
import requests
from io import BytesIO
from window_tracker import WindowTracker
from vision_helper import VisionHelper
from input_handler import InputHandler
from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_PROMPT,
    PUSHOVER_API_TOKEN,
    PUSHOVER_USER_KEY,
)

class CaptchaSolver:
    def __init__(self, tracker: WindowTracker, vision: VisionHelper, input_handler: InputHandler):
        self.tracker = tracker
        self.vision = vision
        self.inputs = input_handler
        self.pushover_user = PUSHOVER_USER_KEY
        self.pushover_token = PUSHOVER_API_TOKEN

    def send_pushover(self, text: str):
        """Sends a notification to the user's mobile device via Pushover."""
        try:
            data = {
                "token": self.pushover_token,
                "user": self.pushover_user,
                "title": "Minecraft Bot Captcha",
                "message": text,
                "priority": 1,
            }
            r = requests.post("https://api.pushover.net/1/messages.json", data=data, timeout=5)
            if r.status_code == 200:
                print("📱 Pushover notification sent.")
            else:
                print(f"⚠️ Pushover failed: {r.status_code} - {r.text}")
        except Exception as e:
            print(f"⚠️ Pushover error: {e}")

    def solve_captcha(self) -> bool:
        """
        Attempts to solve the currently open chest GUI captcha.
        Returns True if solved successfully, False otherwise.
        """
        print("🔍 Captcha container detected! Analyzing screenshot via Gemini API...")
        
        # Take screenshot of the game window
        try:
            window_img = self.vision.capture_window()
        except Exception as e:
            print(f"⚠️ Failed to capture window screenshot: {e}")
            return False

        # Convert image to base64 bytes
        buffered = BytesIO()
        window_img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Call Gemini API via raw HTTP POST request to avoid library SDK issues
        api_key = GEMINI_API_KEY
        if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
            print("⚠️ GEMINI_API_KEY is not configured. Falling back to manual alert.")
            self.send_pushover("Captcha popped up but GEMINI_API_KEY is not configured! Please solve manually.")
            return False

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": GEMINI_PROMPT},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": img_b64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            if response.status_code != 200:
                print(f"⚠️ Gemini API returned status code {response.status_code}: {response.text}")
                self.send_pushover("Gemini API call failed. Please solve captcha manually.")
                return False

            res_json = response.json()
            # Extract generated response text
            resp_text = res_json['candidates'][0]['content']['parts'][0]['text']
            print(f"🤖 Gemini response raw output: {resp_text.strip()}")

            data = json.loads(resp_text)
            is_captcha = data.get("is_captcha", False)
            target_item = data.get("target_item", "Unknown")

            if not is_captcha:
                print("ℹ️ Gemini identified this screen as NOT a captcha GUI.")
                return False

            click_coords = data.get("click_coords")
            if not click_coords or "x" not in click_coords or "y" not in click_coords:
                print("⚠️ Gemini identified captcha but did not return valid click coordinates.")
                self.send_pushover(f"Gemini identified captcha for '{target_item}' but coordinates were missing.")
                return False

            # Translate screenshot window-relative click coordinates to absolute screen coordinates
            wx, wy, ww, wh = self.tracker.get_rect()
            abs_x = wx + int(click_coords["x"])
            abs_y = wy + int(click_coords["y"])

            print(f"🎯 Captcha Target Identified: '{target_item}'")
            print(f"🖱️ Clicking item slot at window pos: ({click_coords['x']}, {click_coords['y']}) -> Screen pos: ({abs_x}, {abs_y})")

            # Click the target item
            self.inputs.drag_mouse_absolute(abs_x, abs_y)
            time.sleep(1.0)
            
            # Close GUI
            self.inputs.press_key("escape")
            time.sleep(0.5)
            return True

        except Exception as e:
            print(f"⚠️ Error during Gemini captcha resolution: {e}")
            self.send_pushover("Exception raised during captcha solving. Please solve manually.")
            return False
