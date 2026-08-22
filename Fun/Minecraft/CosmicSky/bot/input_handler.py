# input_handler.py
import ctypes
import time
import random
import pyautogui
import keyboard

# --- Windows API input structures for SendInput ---
c_ulong = ctypes.c_ulong
c_long = ctypes.c_long

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", c_long),
        ("dy", c_long),
        ("mouseData", c_ulong),
        ("dwFlags", c_ulong),
        ("time", c_ulong),
        ("dwExtraInfo", ctypes.POINTER(c_ulong))
    ]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", c_ulong),
        ("time", c_ulong),
        ("dwExtraInfo", ctypes.POINTER(c_ulong))
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort)
    ]

class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT)
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", c_ulong),
        ("union", INPUT_UNION)
    ]

# Win32 Constants
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

# Scan codes for standard keys (useful for holding movement keys in Minecraft)
SCAN_CODES = {
    'w': 0x11,
    'a': 0x1E,
    's': 0x1F,
    'd': 0x20,
    'space': 0x39,
    'shift': 0x2A
}

KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP = 0x0002

class InputHandler:
    def __init__(self):
        pass

    def move_mouse_relative(self, dx: int, dy: int):
        """Moves the mouse cursor relatively. Essential for Minecraft camera rotation."""
        extra = ctypes.c_ulong(0)
        ii_ = INPUT_UNION()
        ii_.mi = MOUSEINPUT(dx, dy, 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))
        command = INPUT(INPUT_MOUSE, ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

    def mouse_click(self, button="left", hold_duration=0.05):
        """Simulates a mouse click using SendInput."""
        extra = ctypes.c_ulong(0)
        ii_ = INPUT_UNION()
        
        if button == "left":
            down_flag = MOUSEEVENTF_LEFTDOWN
            up_flag = MOUSEEVENTF_LEFTUP
        else:
            down_flag = MOUSEEVENTF_RIGHTDOWN
            up_flag = MOUSEEVENTF_RIGHTUP

        # Press down
        ii_.mi = MOUSEINPUT(0, 0, 0, down_flag, 0, ctypes.pointer(extra))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(INPUT(INPUT_MOUSE, ii_)), ctypes.sizeof(INPUT))
        
        # Click duration delay
        time.sleep(hold_duration)
        
        # Release up
        ii_.mi = MOUSEINPUT(0, 0, 0, up_flag, 0, ctypes.pointer(extra))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(INPUT(INPUT_MOUSE, ii_)), ctypes.sizeof(INPUT))

    def click_human_jitter(self, button="left", min_delay=0.05, max_delay=0.15):
        """Clicks once, then sleeps a randomized delay to simulate human clicking pattern."""
        self.mouse_click(button)
        jitter = random.uniform(min_delay, max_delay)
        time.sleep(jitter)

    def drag_mouse_absolute(self, target_x: int, target_y: int):
        """Moves the mouse to an absolute coordinate and clicks (used for GUI inventories/captchas)."""
        pyautogui.moveTo(target_x, target_y)
        time.sleep(0.1)
        pyautogui.click()

    def press_key(self, key: str):
        """Quickly presses a key (useful for hotbar numbers)."""
        pyautogui.press(key)

    def type_command(self, text: str):
        """Types a chat command in Minecraft (opens chat, types, and sends)."""
        # Close any open GUI first
        pyautogui.press("escape")
        time.sleep(0.2)
        # Open chat
        pyautogui.press("t")
        time.sleep(0.2)
        # Type content
        pyautogui.write(text, interval=0.01)
        time.sleep(0.1)
        # Press Enter
        pyautogui.press("enter")
        time.sleep(0.2)

    def hold_movement_key(self, key: str, duration: float):
        """Holds a movement key using scan codes. Important for in-game movement."""
        if key not in SCAN_CODES:
            # Fallback to keyboard module if key scan code not mapped
            keyboard.press(key)
            time.sleep(duration)
            keyboard.release(key)
            return

        scan_code = SCAN_CODES[key]
        extra = ctypes.c_ulong(0)
        
        # Press key
        ii_ = INPUT_UNION()
        ii_.ki = KEYBDINPUT(0, scan_code, KEYEVENTF_SCANCODE, 0, ctypes.pointer(extra))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(INPUT(INPUT_KEYBOARD, ii_)), ctypes.sizeof(INPUT))
        
        time.sleep(duration)
        
        # Release key
        ii_ = INPUT_UNION()
        ii_.ki = KEYBDINPUT(0, scan_code, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, ctypes.pointer(extra))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(INPUT(INPUT_KEYBOARD, ii_)), ctypes.sizeof(INPUT))
