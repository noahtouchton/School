import sys
import asyncio
import os

try:
    from iaqualink.client import AqualinkClient
except ImportError:
    try:
        from iaqualink import AqualinkClient
    except ImportError:
        from iaqualink.system import AqualinkSystem as AqualinkClient

USERNAME = "cwtouch@bellsouth.net"
PASSWORD = "zynfuhXogzop6qiwqu!"  # <-- Paste your password back in!

STATE_FILE = "/home/hoobs/system_state.txt"

def get_system_state():
    if not os.path.exists(STATE_FILE):
        return "OFF"
    with open(STATE_FILE, "r") as f:
        return f.read().strip()

def set_system_state(state):
    with open(STATE_FILE, "w") as f:
        f.write(state.upper())

async def master_control():
    if len(sys.argv) < 3:
        return

    device = sys.argv[1].lower()   
    action = sys.argv[2].lower()   
    value = sys.argv[3].upper() if len(sys.argv) > 3 else None

    # --- 1. THE MASTER KILL SWITCH ---
    if device == "system":
        if action == "get":
            print(get_system_state())
        elif action == "set" and value:
            set_system_state(value)
        return

    # --- 2. API BLOCKER ---
    if get_system_state() == "OFF" and action == "set":
        return

    # --- CLOUD CONNECTION ---
    async with AqualinkClient(USERNAME, PASSWORD) as client:
        systems = await client.get_systems()
        if not systems: return
        system = list(systems.values())[0]
        devices = await system.get_devices()

        # --- 3. THE MACRO: INDEPENDENT THERMOSTAT STATUS ---
        if device == "macro_pump":
            pump = devices.get("pool_pump")
            temp1_hot = devices.get("spa_heater")    # Mapped to Temp 1
            temp2_cold = devices.get("pool_heater")  # Mapped to Temp 2
            
            if action == "get":
                # Now checks if Temp 1 is ON, ignoring the pump's cooldown state
                print("ON" if temp1_hot.is_on else "OFF")
            
            elif action == "set" and value == "ON":
                await pump.turn_on()
                await temp1_hot.turn_on()
                await temp2_cold.turn_off()
            
            elif action == "set" and value == "OFF":
                await temp1_hot.turn_off()
                await temp2_cold.turn_on()
                # Pump is intentionally untouched here for cooldown

        # --- 4. AIR BLOWER ---
        elif device == "blower":
            blower = devices.get("aux_1")
            if action == "get":
                print("ON" if blower.is_on else "OFF")
            elif action == "set" and value:
                if value == "ON": await blower.turn_on()
                else: await blower.turn_off()

        # --- 5. SPA LIGHT ---
        elif device == "light":
            light = devices.get("aux_2")
            if action == "get":
                print("ON" if light.is_on else "OFF")
            elif action == "set" and value:
                if value == "ON": await light.turn_on()
                else: await light.turn_off()

if __name__ == "__main__":
    asyncio.run(master_control())
