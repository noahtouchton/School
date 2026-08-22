# main_controller.py
import time
import sys
import keyboard
from window_tracker import WindowTracker
from input_handler import InputHandler
from vision_helper import VisionHelper
from fishing_mode import FishingMode
from slaying_mode import SlayingMode
from mining_mode import MiningMode
from captcha_solver import CaptchaSolver
from config import EXIT_KEY, PAUSE_KEY, SELL_CHAT_COMMAND, SELL_INTERVAL_SEC

# State Machine Modes
MODE_PAUSED = "PAUSED"
MODE_SLAYING = "SLAYING"
MODE_MINING = "MINING"
MODE_FISHING = "FISHING"

class MainController:
    def __init__(self):
        print("🤖 Initializing Minecraft Skyblock Automation Controller...")
        self.tracker = WindowTracker()
        self.inputs = InputHandler()
        self.vision = VisionHelper(self.tracker)

        # Mode Handlers
        self.fishing = FishingMode(self.tracker, self.inputs, self.vision)
        self.slaying = SlayingMode(self.tracker, self.inputs, self.vision)
        self.mining = MiningMode(self.tracker, self.inputs, self.vision)
        self.captcha = CaptchaSolver(self.tracker, self.vision, self.inputs)

        # State Variables
        self.active_mode = MODE_PAUSED
        self.is_running = True
        self.tick_counter = 0
        self.last_sell_time = time.time()
        self.expecting_gui = False  # Set to True when bot purposefully opens chat/menus

    def setup_hotkeys(self):
        """Binds hotkeys for manually switching modes."""
        print("\n⌨️ Hotkey Bindings:")
        print(f"  [{EXIT_KEY}]  -> Emergency Stop")
        print(f"  [{PAUSE_KEY}]  -> Pause/Unpause")
        print("  [f5] -> Slaying Mode")
        print("  [f6] -> Mining Mode")
        print("  [f7] -> Fishing Mode\n")

        keyboard.add_hotkey(EXIT_KEY, self.emergency_stop)
        keyboard.add_hotkey(PAUSE_KEY, self.toggle_pause)
        keyboard.add_hotkey("f5", lambda: self.set_mode(MODE_SLAYING))
        keyboard.add_hotkey("f6", lambda: self.set_mode(MODE_MINING))
        keyboard.add_hotkey("f7", lambda: self.set_mode(MODE_FISHING))

    def set_mode(self, new_mode: str):
        """Transitions bot state, calling start/stop hooks on behavior modules."""
        if self.active_mode == new_mode:
            return
            
        print(f"🔄 Mode transition: {self.active_mode} ➡️ {new_mode}")
        
        # Stop current mode handlers if needed
        if self.active_mode == MODE_MINING:
            self.mining.stop()

        self.active_mode = new_mode

        # Start new mode handlers
        if new_mode == MODE_SLAYING:
            self.slaying.start()
        elif new_mode == MODE_MINING:
            self.mining.start()
        elif new_mode == MODE_FISHING:
            self.fishing.start()

    def toggle_pause(self):
        if self.active_mode == MODE_PAUSED:
            self.set_mode(MODE_SLAYING)  # Resume with slaying or default
        else:
            self.set_mode(MODE_PAUSED)

    def emergency_stop(self):
        """Immediately stops the bot and releases any held keys."""
        print("\n🚨 EMERGENCY STOP TRIGGERED. Exiting...")
        self.is_running = False
        if self.active_mode == MODE_MINING:
            self.mining.stop()
        # Safety key release hook
        for key in ['w', 'a', 's', 'd']:
            keyboard.release(key)
        sys.exit(0)

    def check_for_captcha(self) -> bool:
        """
        Checks if a GUI is open unexpectedly. If so, triggers captcha resolution.
        Returns True if a captcha was active, pausing standard loop execution.
        """
        if self.tracker.is_gui_open():
            # If the GUI is open but we didn't expect it, it is likely a captcha puzzle
            if not self.expecting_gui:
                print("⚠️ Unexpected GUI opened! Initiating Captcha Solver...")
                # Stop active movements to be safe
                if self.active_mode == MODE_MINING:
                    self.mining.stop()

                # Call captcha solver
                solved = self.captcha.solve_captcha()
                
                if solved:
                    print("✅ Captcha resolved! Resuming state.")
                    # Restart current mode hooks
                    if self.active_mode == MODE_SLAYING:
                        self.slaying.start()
                    elif self.active_mode == MODE_MINING:
                        self.mining.start()
                    elif self.active_mode == MODE_FISHING:
                        self.fishing.start()
                else:
                    print("❌ Captcha solver failed or manual intervention requested. Pausing bot.")
                    self.set_mode(MODE_PAUSED)
                return True
        return False

    def perform_auto_sell(self):
        """Executes the auto-sell chat sequence periodically."""
        current_time = time.time()
        if current_time - self.last_sell_time >= SELL_INTERVAL_SEC:
            print("💰 Performing scheduled auto-sell...")
            self.expecting_gui = True
            
            # Stop active movement to avoid opening chat while running/mining
            if self.active_mode == MODE_MINING:
                self.mining.stop()

            # Execute selling command
            self.inputs.type_command(SELL_CHAT_COMMAND)
            self.last_sell_time = current_time
            time.sleep(1.0) # Wait for chat to process
            
            # Restart current mode hooks
            if self.active_mode == MODE_SLAYING:
                self.slaying.start()
            elif self.active_mode == MODE_MINING:
                self.mining.start()
            elif self.active_mode == MODE_FISHING:
                self.fishing.start()
                
            self.expecting_gui = False

    def run(self):
        """Main orchestrator execution loop."""
        self.setup_hotkeys()
        print("🚀 Controller is running. Switch to the Minecraft window!")
        print("Bot is currently PAUSED. Press F5, F6, or F7 to begin a mode.")
        
        while self.is_running:
            self.tick_counter += 1
            
            # 1. Check for emergency quit key
            if keyboard.is_pressed(EXIT_KEY):
                self.emergency_stop()

            # 2. Skip logic if paused
            if self.active_mode == MODE_PAUSED:
                time.sleep(0.1)
                continue

            # 3. Check for unexpected captcha GUIs
            if self.check_for_captcha():
                time.sleep(1.0)
                continue

            # 4. Check for auto-sell intervals
            self.perform_auto_sell()

            # 5. Execute tick behavior based on mode
            try:
                if self.active_mode == MODE_SLAYING:
                    self.slaying.tick(self.tick_counter)
                elif self.active_mode == MODE_MINING:
                    self.mining.tick(self.tick_counter)
                elif self.active_mode == MODE_FISHING:
                    self.fishing.tick(self.tick_counter)
            except Exception as e:
                print(f"⚠️ Error running active mode tick: {e}")
                self.set_mode(MODE_PAUSED)

            # Sleep briefly to regulate CPU cycles (approx 20 ticks per second)
            time.sleep(0.05)

def main():
    try:
        controller = MainController()
        controller.run()
    except KeyboardInterrupt:
        print("\n👋 Bot controller terminated manually via keyboard.")
    except Exception as e:
        print(f"\n❌ Critical controller error: {e}")

if __name__ == "__main__":
    main()
