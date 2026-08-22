# mining_mode.py
import time
from window_tracker import WindowTracker
from input_handler import InputHandler
from vision_helper import VisionHelper
from config import MINING_PATH

# Mining States
STATE_IDLE = "IDLE"
STATE_AIMING = "AIMING"
STATE_MINING = "MINING"
STATE_MOVING = "MOVING"

class MiningMode:
    def __init__(self, tracker: WindowTracker, input_handler: InputHandler, vision: VisionHelper):
        self.tracker = tracker
        self.inputs = input_handler
        self.vision = vision
        
        self.path = MINING_PATH
        self.current_step_index = 0
        
        # State Machine Variables
        self.state = STATE_IDLE
        self.state_start_time = 0.0
        self.state_duration = 0.0

    def start(self):
        """Initializes the mining run."""
        print("⛏️ Starting Mining Mode...")
        self.current_step_index = 0
        self.enter_state(STATE_AIMING)

    def stop(self):
        """Cleans up inputs on stop."""
        self.inputs.inputs = None  # Ensure mouse release
        self.state = STATE_IDLE

    def enter_state(self, new_state: str):
        """Handles state transition side effects and timing."""
        self.state = new_state
        self.state_start_time = time.time()
        
        step_config = self.path[self.current_step_index]

        if new_state == STATE_AIMING:
            # Rotate camera relative to target yaw/pitch
            yaw = step_config.get("yaw", 0)
            pitch = step_config.get("pitch", 0)
            print(f"⛏️ Mining Step {self.current_step_index + 1}/{len(self.path)} - Aiming: yaw={yaw}, pitch={pitch}")
            self.inputs.move_mouse_relative(yaw, pitch)
            self.state_duration = 0.3  # Short pause to stabilize camera

        elif new_state == STATE_MINING:
            # Press mouse left click down to mine
            print(f"⛏️ Mining block...")
            # We trigger a continuous mouse down. In our input handler, we can simulate
            # holding by calling mouse down. Since our input_handler does click-down and click-up,
            # we will trigger a mouse click hold.
            # To be simple and robust in raw input, we can hold the left click down.
            # We will use Win32 SendInput to press Left Mouse Down.
            self.send_mouse_down(button="left")
            self.state_duration = step_config.get("mine_duration", 1.5)

        elif new_state == STATE_MOVING:
            # Release mouse left click, then press movement key
            self.send_mouse_up(button="left")
            move_dur = step_config.get("move_duration", 0.5)
            print(f"⛏️ Moving forward for {move_dur}s...")
            # Hold forward 'w' key
            self.inputs.hold_movement_key('w', move_dur)
            self.state_duration = move_dur

    def send_mouse_down(self, button="left"):
        """Sends raw Mouse Down event."""
        import ctypes
        from input_handler import INPUT, INPUT_UNION, MOUSEINPUT, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_RIGHTDOWN
        extra = ctypes.c_ulong(0)
        ii_ = INPUT_UNION()
        flag = MOUSEEVENTF_LEFTDOWN if button == "left" else MOUSEEVENTF_RIGHTDOWN
        ii_.mi = MOUSEINPUT(0, 0, 0, flag, 0, ctypes.pointer(extra))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(INPUT(0, ii_)), ctypes.sizeof(INPUT))

    def send_mouse_up(self, button="left"):
        """Sends raw Mouse Up event."""
        import ctypes
        from input_handler import INPUT, INPUT_UNION, MOUSEINPUT, MOUSEEVENTF_LEFTUP, MOUSEEVENTF_RIGHTUP
        extra = ctypes.c_ulong(0)
        ii_ = INPUT_UNION()
        flag = MOUSEEVENTF_LEFTUP if button == "left" else MOUSEEVENTF_RIGHTUP
        ii_.mi = MOUSEINPUT(0, 0, 0, flag, 0, ctypes.pointer(extra))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(INPUT(0, ii_)), ctypes.sizeof(INPUT))

    def tick(self, counter: int):
        """
        Called on every main loop tick. Checks state duration and advances the mining process.
        """
        if self.state == STATE_IDLE:
            return

        elapsed = time.time() - self.state_start_time

        if elapsed >= self.state_duration:
            # Advance State Machine
            if self.state == STATE_AIMING:
                self.enter_state(STATE_MINING)
            elif self.state == STATE_MINING:
                self.enter_state(STATE_MOVING)
            elif self.state == STATE_MOVING:
                # Completed this node, advance to next node
                self.current_step_index = (self.current_step_index + 1) % len(self.path)
                self.enter_state(STATE_AIMING)
