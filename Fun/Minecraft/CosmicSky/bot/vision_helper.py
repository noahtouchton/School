# vision_helper.py
import os
import pyautogui
from PIL import Image, ImageDraw
from window_tracker import WindowTracker

class VisionHelper:
    def __init__(self, tracker: WindowTracker):
        self.tracker = tracker

    def capture_screen(self) -> Image.Image:
        """Captures a fullscreen screenshot and returns a RGB PIL Image."""
        return pyautogui.screenshot().convert("RGB")

    def capture_window(self) -> Image.Image:
        """Captures only the Minecraft game window client area."""
        wx, wy, ww, wh = self.tracker.get_rect()
        return pyautogui.screenshot(region=(wx, wy, ww, wh)).convert("RGB")

    def capture_relative_region(self, rect_rel: tuple[float, float, float, float]) -> Image.Image:
        """
        Takes a screenshot of a region defined by relative coordinates.
        rect_rel: (x1_rel, y1_rel, x2_rel, y2_rel) e.g., (0.45, 0.45, 0.55, 0.55)
        """
        x1, y1, x2, y2 = self.tracker.rel_to_abs_rect(*rect_rel)
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        return pyautogui.screenshot(region=(x1, y1, w, h)).convert("RGB")

    def get_relative_pixel_color(self, point_rel: tuple[float, float]) -> tuple[int, int, int]:
        """Gets the RGB color of a pixel at a relative coordinate."""
        x_abs, y_abs = self.tracker.rel_to_abs_point(*point_rel)
        # Avoid full screen capture if possible by capturing a 1x1 region
        img = pyautogui.screenshot(region=(x_abs, y_abs, 1, 1)).convert("RGB")
        return img.getpixel((0, 0))

    @staticmethod
    def rgb_close(px: tuple[int, int, int], tgt: tuple[int, int, int], margin: int) -> bool:
        """Compares two RGB pixels with a tolerance margin."""
        return all(abs(int(px[i]) - int(tgt[i])) <= int(margin) for i in range(3))

    def save_debug_overlay(self, points: dict[str, tuple[float, float]], rects: dict[str, tuple[float, float, float, float]], output_path="config_preview.png"):
        """
        Saves a screenshot highlighting all relative points and rectangles.
        Helps visually confirm that relative coordinates align with game UI elements.
        """
        win_rect = self.tracker.get_rect()
        wx, wy, ww, wh = win_rect
        window_img = self.capture_window()
        draw = ImageDraw.Draw(window_img)

        # Draw relative crop rectangles
        for name, rect_rel in rects.items():
            x1_abs, y1_abs, x2_abs, y2_abs = self.tracker.rel_to_abs_rect(*rect_rel)
            # Shift to window-local coordinates
            x1, y1 = x1_abs - wx, y1_abs - wy
            x2, y2 = x2_abs - wx, y2_abs - wy
            
            draw.rectangle([x1, y1, x2, y2], outline="blue" if "crop" in name else "green", width=3)
            draw.text((x1 + 5, y1 + 5), name, fill="blue" if "crop" in name else "green")

        # Draw relative single points
        for name, point_rel in points.items():
            x_abs, y_abs = self.tracker.rel_to_abs_point(*point_rel)
            x, y = x_abs - wx, y_abs - wy
            r = 6
            # Draw crosshair + circle
            draw.line((x - 10, y, x + 10, y), fill="red", width=2)
            draw.line((x, y - 10, x, y + 10), fill="red", width=2)
            draw.ellipse([x - r, y - r, x + r, y + r], outline="red", width=2)
            draw.text((x + 8, y + 8), name, fill="red")

        window_img.save(output_path)
        print(f"📸 Debug overlay saved to: {os.path.abspath(output_path)}")
