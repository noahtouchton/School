# window_tracker.py
import ctypes
from ctypes import wintypes
import pygetwindow as gw
from config import WINDOW_TITLES, DPI_AWARE

# Configure DPI Awareness
if DPI_AWARE:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception as e:
        print(f"⚠️ DPI Awareness configuration warning: {e}")

# Windows Cursor Info Structures for Ctypes
class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

class CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hCursor", wintypes.HANDLE),
        ("ptScreenPos", POINT)
    ]

CURSOR_SHOWING = 0x00000001

class WindowTracker:
    def __init__(self):
        self.win_rect = None
        self.update_window()

    def update_window(self):
        """Locates the Minecraft game window and updates the cached rect (x, y, w, h)."""
        for title in WINDOW_TITLES:
            all_windows = gw.getWindowsWithTitle(title)
            for win in all_windows:
                t = win.title.lower()
                # Exclude launchers and development IDEs
                if "launcher" in t or "pycharm" in t or "vscode" in t:
                    continue
                
                # Check active game states in title (e.g. Multiplayer, Singleplayer, server names)
                # CosmicSky typically has "Minecraft" in title
                if "minecraft" in t or "cosmic" in t or any(kw in t for kw in ["multiplayer", "singleplayer", "server"]):
                    self.win_rect = (win.left, win.top, win.width, win.height)
                    return self.win_rect
        
        # Fallback if no specific window found, try to locate any window starting with Minecraft
        all_windows = gw.getWindowsWithTitle("Minecraft")
        if all_windows:
            win = all_windows[0]
            self.win_rect = (win.left, win.top, win.width, win.height)
            return self.win_rect
            
        raise RuntimeError("Minecraft game window not found. Please make sure the game is running.")

    def get_rect(self):
        """Returns the current window bounds, updating them dynamically."""
        self.update_window()
        return self.win_rect

    def rel_to_abs_point(self, xr: float, yr: float) -> tuple[int, int]:
        """Translates relative coordinates (0.0 to 1.0) to absolute screen pixel positions."""
        wx, wy, ww, wh = self.get_rect()
        return (int(wx + xr * ww), int(wy + yr * wh))

    def rel_to_abs_rect(self, x1r: float, y1r: float, x2r: float, y2r: float) -> tuple[int, int, int, int]:
        """Translates relative crop coordinates to absolute screen rectangle coordinates."""
        wx, wy, ww, wh = self.get_rect()
        return (
            int(wx + x1r * ww), int(wy + y1r * wh),
            int(wx + x2r * ww), int(wy + y2r * wh)
        )

    @staticmethod
    def is_gui_open() -> bool:
        """
        Detects if a Minecraft GUI (inventory, chest, captcha, chat, pause menu) is open.
        When Minecraft captures the mouse (first-person), the cursor is hidden (flags = 0).
        When a GUI is open, the mouse cursor is visible (flags = CURSOR_SHOWING).
        """
        cursor_info = CURSORINFO()
        cursor_info.cbSize = ctypes.sizeof(CURSORINFO)
        if ctypes.windll.user32.GetCursorInfo(ctypes.byref(cursor_info)):
            return (cursor_info.flags & CURSOR_SHOWING) != 0
        return False
