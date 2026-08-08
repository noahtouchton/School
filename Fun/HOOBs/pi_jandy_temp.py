import sys
import asyncio
import os
import time
import json

try:
    from iaqualink.client import AqualinkClient
except ImportError:
    from iaqualink import AqualinkClient

USERNAME = "cwtouch@bellsouth.net"
PASSWORD = "zynfuhXogzop6qiwqu!"

CACHE_FILE = "/tmp/jandy_cache.json"
LOCK_FILE = "/tmp/jandy_api.lock"

def fahrenheit_to_celsius(f_val):
    try:
        f = float(f_val)
        c = (f - 32.0) * 5.0 / 9.0
        return round(c, 1)
    except Exception:
        return 20.0  # Fallback default room temp

def celsius_to_fahrenheit(c_val):
    try:
        c = float(c_val)
        f = (c * 9.0 / 5.0) + 32.0
        f_int = int(round(f))
        # Cap at Jandy safety limit of 104 to prevent API errors
        return min(f_int, 104)
    except Exception:
        return 100  # Fallback default target

async def update_cache_from_api():
    # Acquire lock. If lock is held by another process and is fresh, wait.
    for _ in range(15):  # wait up to 15 seconds
        if os.path.exists(LOCK_FILE):
            try:
                lock_time = os.path.getmtime(LOCK_FILE)
                if time.time() - lock_time < 15:
                    await asyncio.sleep(1)
                    continue
            except OSError:
                pass
        break
        
    # Write lock file
    try:
        with open(LOCK_FILE, "w") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

    try:
        # Re-check cache in case another process just updated it while we were waiting
        if os.path.exists(CACHE_FILE):
            mtime = os.path.getmtime(CACHE_FILE)
            if time.time() - mtime < 30:
                return  # Cache is now fresh
                
        async with AqualinkClient(USERNAME, PASSWORD) as client:
            systems = await client.get_systems()
            if not systems:
                return
            system = list(systems.values())[0]
            devices = await system.get_devices()
            
            cache_data = {
                "timestamp": time.time(),
                "spa_temp": devices.get("spa_temp").state if devices.get("spa_temp") else "",
                "pool_temp": devices.get("pool_temp").state if devices.get("pool_temp") else "",
                "air_temp": devices.get("air_temp").state if devices.get("air_temp") else "",
                "spa_set_point": devices.get("spa_set_point").state if devices.get("spa_set_point") else "100",
                "spa_heater": devices.get("spa_heater").state if devices.get("spa_heater") else "0"
            }
            
            with open(CACHE_FILE, "w") as f:
                json.dump(cache_data, f)
    except Exception as e:
        sys.stderr.write(f"API Error: {e}\n")
    finally:
        # Release lock
        try:
            os.remove(LOCK_FILE)
        except OSError:
            pass

async def get_device_data():
    cache_fresh = False
    if os.path.exists(CACHE_FILE):
        try:
            mtime = os.path.getmtime(CACHE_FILE)
            if time.time() - mtime < 30:
                cache_fresh = True
        except OSError:
            pass
            
    if not cache_fresh:
        await update_cache_from_api()
        
    # Read cache
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        # Fallback empty structure
        return {
            "spa_temp": "",
            "pool_temp": "",
            "air_temp": "",
            "spa_set_point": "100",
            "spa_heater": "0"
        }

async def temp_control():
    if len(sys.argv) < 4:
        return
    
    action = sys.argv[1] # Get/Set
    char = sys.argv[3]   # CurrentTemperature, TargetTemperature, etc.
    value = sys.argv[4] if len(sys.argv) > 4 else None

    if action == "Get":
        cache = await get_device_data()
        
        if char == "CurrentTemperature":
            # Get actual temperature, fall back to air temp if pool/spa temp are empty (e.g. pumps off)
            current_temp_f = ""
            if cache.get("spa_temp"):
                current_temp_f = cache.get("spa_temp")
            elif cache.get("pool_temp"):
                current_temp_f = cache.get("pool_temp")
                
            if not current_temp_f:
                if cache.get("air_temp"):
                    current_temp_f = cache.get("air_temp")
                else:
                    current_temp_f = "70"
            
            print(fahrenheit_to_celsius(current_temp_f))

        elif char == "TargetTemperature":
            target_f = cache.get("spa_set_point", "100")
            print(fahrenheit_to_celsius(target_f))

        elif char == "CurrentHeatingCoolingState" or char == "TargetHeatingCoolingState":
            # Always return "1" (Heat) so the thermostat dial is always active/orange in HomeKit
            print("1")

        elif char == "TemperatureDisplayUnits":
            print("1") # 0 = Celsius, 1 = Fahrenheit

    elif action == "Set":
        if char == "TargetTemperature":
            # Direct API call to update temperature
            target_f = celsius_to_fahrenheit(value)
            async with AqualinkClient(USERNAME, PASSWORD) as client:
                systems = await client.get_systems()
                if systems:
                    system = list(systems.values())[0]
                    devices = await system.get_devices()
                    hot_setpoint = devices.get("spa_set_point")
                    if hot_setpoint:
                        await hot_setpoint.set_temperature(target_f)
            
            # Invalidate cache to force reload on next get
            try:
                os.remove(CACHE_FILE)
            except OSError:
                pass

if __name__ == "__main__":
    asyncio.run(temp_control())
