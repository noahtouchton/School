# fishing_mode.py
import time
import os
from window_tracker import WindowTracker
from input_handler import InputHandler
from vision_helper import VisionHelper
from config import FISHING_POLE_SLOT, FISHING_CROP_REL, FISH_DETECTOR_MODEL_PATH

# Optional deep learning imports for the ResNet model
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision.models import resnet18
    TORCH_AVAILABLE = True
except ImportError:
    pass

class FishingMode:
    def __init__(self, tracker: WindowTracker, input_handler: InputHandler, vision: VisionHelper):
        self.tracker = tracker
        self.inputs = input_handler
        self.vision = vision
        self.model = None
        self.last_recast_time = time.time()
        self.is_cast = False
        
        self._init_ml_model()

    def _init_ml_model(self):
        """Initializes and loads the ResNet18 fish detector model if available."""
        if not TORCH_AVAILABLE:
            print("ℹ️ PyTorch/Torchvision is not installed. Fishing will use color-based fallback.")
            return

        if not os.path.exists(FISH_DETECTOR_MODEL_PATH):
            print(f"ℹ️ Model file '{FISH_DETECTOR_MODEL_PATH}' not found. Fishing will use color-based fallback.")
            return

        try:
            print("🌀 Loading ResNet18 fish detector ML model...")
            self.model = resnet18(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)
            self.model.load_state_dict(torch.load(FISH_DETECTOR_MODEL_PATH, map_location=torch.device("cpu")))
            self.model.eval()
            
            # Setup image transforms
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
            print("✅ ML fish detector model loaded successfully.")
        except Exception as e:
            print(f"⚠️ Error loading fishing ML model: {e}. Falling back to color-based detection.")
            self.model = None

    def start(self):
        """Prepares the bot for fishing."""
        print("🎣 Starting Fishing Mode...")
        self.inputs.press_key(FISHING_POLE_SLOT)
        time.sleep(0.5)
        self.cast()

    def cast(self):
        """Casts the fishing line."""
        print("🎣 Casting rod...")
        self.inputs.mouse_click("right")
        self.last_recast_time = time.time()
        self.is_cast = True
        time.sleep(1.0)  # Wait for cast animation to settle

    def reel_in(self):
        """Reels in the line and recasts."""
        print("🐟 Fish detected! Reeling in...")
        self.inputs.mouse_click("right")
        time.sleep(0.2)
        # Recast
        self.cast()

    def check_fish_ml(self) -> bool:
        """Runs the ResNet classifier on the center screen crop."""
        if self.model is None or not TORCH_AVAILABLE:
            return False

        try:
            # Capture the center crop region of the screen
            img = self.vision.capture_relative_region(FISHING_CROP_REL)
            tensor = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(tensor)
                prediction = torch.argmax(outputs, dim=1).item()
            return prediction == 1  # 1 = fish/particles present
        except Exception as e:
            print(f"⚠️ Error running ML prediction: {e}")
            return False

    def check_fish_color(self) -> bool:
        """
        Fallback color check: scans the center crop region for high concentration of red particles.
        """
        try:
            img = self.vision.capture_relative_region(FISHING_CROP_REL)
            w, h = img.size
            # CosmicSky fishing particles are bright reddish (approx 252, 84, 84)
            target_red = (252, 84, 84)
            red_pixels = 0
            
            for x in range(0, w, 2):  # Step by 2 to speed up scan
                for y in range(0, h, 2):
                    pixel = img.getpixel((x, y))
                    if self.vision.rgb_close(pixel, target_red, margin=20):
                        red_pixels += 1
                        
            # If we find enough reddish particle pixels, register a trigger
            return red_pixels >= 4
        except Exception as e:
            print(f"⚠️ Error running color-based fish check: {e}")
            return False

    def tick(self, counter: int):
        """
        Called on every loop cycle. Runs detection and handles timing updates.
        """
        if not self.is_cast:
            self.cast()
            return

        # Check for fish
        detected = False
        if self.model is not None:
            detected = self.check_fish_ml()
        else:
            detected = self.check_fish_color()

        if detected:
            self.reel_in()
            return

        # Safety recast (anti-idle / cast state validation)
        # e.g., if hook misses or is idle for 45s, recast
        current_time = time.time()
        if current_time - self.last_recast_time > 45.0:
            print("⏳ Fishing hook idle timeout. Recasting...")
            self.inputs.mouse_click("right") # Reel in if cast
            time.sleep(0.5)
            self.cast()
