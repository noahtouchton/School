# slaying_mode.py
import time
import random
from window_tracker import WindowTracker
from input_handler import InputHandler
from vision_helper import VisionHelper
from config import WEAPON_SLOT, SLAY_TARGETS, SLAY_CLICK_MIN_DELAY, SLAY_CLICK_MAX_DELAY, SLAY_CLICK_COOLDOWN

class SlayingMode:
    def __init__(self, tracker: WindowTracker, input_handler: InputHandler, vision: VisionHelper):
        self.tracker = tracker
        self.inputs = input_handler
        self.vision = vision
        self.target_index = 0
        self.last_target_switch_time = time.time()
        self.clicks_on_current_target = 0
        self.max_clicks_per_target = 10

    def start(self):
        """Prepares the bot for slaying."""
        print("⚔️ Starting Slaying Mode...")
        self.inputs.press_key(WEAPON_SLOT)
        time.sleep(0.5)
        self.target_index = 0
        self.aim_at_target()

    def aim_at_target(self):
        """
        Aims the crosshair at the current configured target.
        In Minecraft, looking around is relative. In a modular skeleton, we simulate
        moving the camera by small increments or using a calibrated yaw/pitch coordinate.
        For first person, we rotate the camera by relative mouse inputs.
        """
        target = SLAY_TARGETS[self.target_index]
        print(f"🎯 Aiming at target {self.target_index}: yaw={target['yaw']}, pitch={target['pitch']}")
        
        # Simulate relative camera rotation based on the configured values
        # (A fully calibrated implementation would calculate the delta yaw/pitch)
        yaw_delta = int(target['yaw'])
        pitch_delta = int(target['pitch'])
        
        # Perform relative mouse move via SendInput
        self.inputs.move_mouse_relative(yaw_delta, pitch_delta)
        self.clicks_on_current_target = 0
        self.last_target_switch_time = time.time()
        
        # Random duration after aiming before hitting
        time.sleep(random.uniform(0.1, 0.25))

    def switch_target(self):
        """Switches to the next mob target in the sequence."""
        self.target_index = (self.target_index + 1) % len(SLAY_TARGETS)
        self.aim_at_target()

    def tick(self, counter: int):
        """
        Performs regular execution steps. Simulates human jitter clicks on targets,
        then switches targets after a set click threshold or timeout.
        """
        # Execute jitter click
        self.inputs.click_human_jitter(
            button="left", 
            min_delay=SLAY_CLICK_MIN_DELAY, 
            max_delay=SLAY_CLICK_MAX_DELAY
        )
        self.clicks_on_current_target += 1

        # Check if we should switch target (after max hits or elapsed duration)
        current_time = time.time()
        should_switch = (
            self.clicks_on_current_target >= self.max_clicks_per_target or
            (current_time - self.last_target_switch_time > 2.5)
        )

        if should_switch:
            time.sleep(SLAY_CLICK_COOLDOWN) # Pause between targets
            self.switch_target()
