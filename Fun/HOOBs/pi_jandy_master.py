import sys
import asyncio

try:
    from iaqualink.client import AqualinkClient
except ImportError:
    try:
        from iaqualink import AqualinkClient
    except ImportError:
        from iaqualink.system import AqualinkSystem as AqualinkClient

USERNAME = "cwtouch@bellsouth.net"
PASSWORD = "zynfuhXogzop6qiwqu!"  # <-- Paste your password back in here!

async def master_control():
    # Example usage: python jandy_master.py spa set ON
    if len(sys.argv) < 3:
        print("Usage: python jandy_master.py <device> <action> [value]")
        return

    device = sys.argv[1].lower()   # air, spa, light, temp1, temp2
    action = sys.argv[2].lower()   # get, set, get_current, get_target
    value = sys.argv[3] if len(sys.argv) > 3 else None

    async with AqualinkClient(USERNAME, PASSWORD) as client:
        systems = await client.get_systems()
        if not systems:
            return
        system = list(systems.values())[0]
        devices = await system.get_devices()

        # -----------------------------------
        # 1. AIR TEMPERATURE SENSOR
        # -----------------------------------
        if device == "air":
            if action == "get":
                print(devices["air_temp"].state)

        # -----------------------------------
        # 2. SPA / POOL PUMP TOGGLE
        # -----------------------------------
        elif device == "spa":
            target = devices.get("pool_pump") # Based on your screenshot
            if action == "get":
                print("ON" if target.is_on else "OFF")
            elif action == "set" and value:
                if value.upper() == "ON": await target.turn_on()
                else: await target.turn_off()

        # -----------------------------------
        # 3. POOL LIGHT TOGGLE
        # -----------------------------------
        elif device == "light":
            # Jandy usually uses pool_light or color_light
            target = devices.get("pool_light") or devices.get("color_light")
            if target:
                if action == "get":
                    print("ON" if target.is_on else "OFF")
                elif action == "set" and value:
                    if value.upper() == "ON": await target.turn_on()
                    else: await target.turn_off()
            else:
                print("Light not found.")

        # -----------------------------------
        # 4. TEMP 1 (HEATER & SETPOINT)
        # -----------------------------------
        elif device == "temp1":
            heater = devices.get("pool_heater")    # The toggle in your screenshot
            setpoint = devices.get("pool_set_point") # The target temp
            current = devices.get("pool_temp")       # The actual water temp
            
            if action == "get_current":
                print(current.state)
            elif action == "get_target":
                print(setpoint.state)
            elif action == "set_target" and value:
                await setpoint.set_temperature(int(value))
            elif action == "get_state":
                print("ON" if heater.is_on else "OFF")
            elif action == "set_state" and value:
                if value.upper() == "ON": await heater.turn_on()
                else: await heater.turn_off()

        # -----------------------------------
        # 5. TEMP 2 (HEATER & SETPOINT)
        # -----------------------------------
        elif device == "temp2":
            heater = devices.get("spa_heater")
            setpoint = devices.get("spa_set_point")
            current = devices.get("pool_temp") # Dux Spa likely uses one primary thermistor
            
            if action == "get_current":
                print(current.state)
            elif action == "get_target":
                print(setpoint.state)
            elif action == "set_target" and value:
                await setpoint.set_temperature(int(value))
            elif action == "get_state":
                print("ON" if heater.is_on else "OFF")
            elif action == "set_state" and value:
                if value.upper() == "ON": await heater.turn_on()
                else: await heater.turn_off()

if __name__ == "__main__":
    asyncio.run(master_control())
