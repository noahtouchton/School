import asyncio

# The working import block we cracked earlier
try:
    from iaqualink.client import AqualinkClient
except ImportError:
    try:
        from iaqualink import AqualinkClient
    except ImportError:
        from iaqualink.system import AqualinkSystem as AqualinkClient

# --- CONFIGURATION ---
USERNAME = "cwtouch@bellsouth.net"
PASSWORD = "zynfuhXogzop6qiwqu!"  # <-- Put your real password back here!
# ---------------------

async def get_air_and_pool_temps():
    async with AqualinkClient(USERNAME, PASSWORD) as client:
        systems = await client.get_systems()
        if not systems:
            print("Error: No systems found.")
            return

        # Grab the Dux Spa system
        system = list(systems.values())[0]
        devices = await system.get_devices()

        # Grab the specific sensors shown at the top of your app
        air_temp = devices.get("air_temp")
        pool_temp = devices.get("pool_temp")

        print("🌡️ --- SENSOR READINGS --- 🌡️")

        if air_temp:
            # .state contains the number (e.g., "47")
            print(f"Air Temp: {air_temp.state}°F")
        else:
            print("Air Temp sensor not found via API.")

        if pool_temp:
            # .state contains the number (e.g., "99")
            print(f"Pool Temp: {pool_temp.state}°F")
        else:
            print("Pool Temp sensor not found via API.")

if __name__ == "__main__":
    asyncio.run(get_air_and_pool_temps())