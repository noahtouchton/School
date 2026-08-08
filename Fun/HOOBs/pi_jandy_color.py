import sys
import asyncio
import os
import time

try:
    from iaqualink.client import AqualinkClient
except ImportError:
    from iaqualink.system import AqualinkSystem as AqualinkClient

USERNAME = "cwtouch@bellsouth.net"
PASSWORD = "zynfuhXogzop6qiwqu!"

# The Crash-Proof Cooldown Lock
LOCK_FILE = "/home/hoobs/color_lock.txt"
STATE_FILE = "/home/hoobs/system_state.txt"

def get_system_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return f.read().strip()
    return "OFF"

async def color_control():
    if len(sys.argv) < 4:
        return
        
    action = sys.argv[1]  # "Get" or "Set"
    char = sys.argv[3]    # "On", "Hue", "Saturation", "Brightness"
    value = sys.argv[4] if len(sys.argv) > 4 else None

    # --- 1. FAKE THE EXTRA SLIDERS ---
    if action == "Get" and char in ["Brightness", "Saturation"]:
        print("100")
        return
    if action == "Set" and char in ["Brightness", "Saturation"]:
        return 

    # --- 2. THE MASTER KILL SWITCH CHECK ---
    if get_system_state() == "OFF" and action == "Set":
        return

    # --- CLOUD CONNECTION ---
    async with AqualinkClient(USERNAME, PASSWORD) as client:
        systems = await client.get_systems()
        if not systems: return
        system = list(systems.values())[0]
        devices = await system.get_devices()
        light = devices.get("aux_2")

        # --- 3. ON/OFF TOGGLE ---
        if char == "On":
            if action == "get":
                print("1" if light.is_on else "0")
            elif action == "set":
                if value == "1" or value.lower() == "true":
                    await light.turn_on()
                else:
                    await light.turn_off()

        # --- 4. THE COLOR WHEEL TRANSLATOR ---
        elif char == "Hue":
            if action == "get":
                print("240") # Default UI position
                return
                
            if action == "set" and value:
                # -- THE 60-SECOND HARDWARE LOCK --
                now = time.time()
                last_time = 0
                if os.path.exists(LOCK_FILE):
                    with open(LOCK_FILE, "r") as f:
                        try:
                            last_time = float(f.read().strip() or 0)
                        except:
                            last_time = 0
                
                if now - last_time < 60:
                    return 
                
                with open(LOCK_FILE, "w") as f:
                    f.write(str(now))

                # -- APPLE TO JANDY TRANSLATOR --
                hue = float(value) % 360
                
                if hue < 30: effect = "Magenta"
                elif hue < 90: effect = "Alpine White"
                elif hue < 150: effect = "Spring Green"
                elif hue < 210: effect = "Sky Blue"
                elif hue < 260: effect = "Cobalt Blue"
                elif hue < 310: effect = "Violet"
                else: effect = "Emerald Rose" # Red spectrum

                try:
                    # UPDATED METHOD FROM DEBUG DUMP
                    await light.set_effect_by_name(effect)
                except Exception:
                    pass

if __name__ == "__main__":
    asyncio.run(color_control())
