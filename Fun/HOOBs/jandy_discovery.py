import asyncio

# The working import block
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

async def discover_devices():
    async with AqualinkClient(USERNAME, PASSWORD) as client:
        systems = await client.get_systems()
        if not systems:
            print("Error: No systems found linked to this account.")
            return

        # Grab your equipment pad
        system = list(systems.values())[0]
        devices = await system.get_devices()

        print("\n🔍 --- JANDY DEVICE DISCOVERY --- 🔍\n")
        
        for name, device in devices.items():
            # Figure out the current state
            state_val = getattr(device, 'state', 'N/A')
            
            # If it's a toggleable switch, check if it's on or off
            if hasattr(device, 'is_on'):
                state_val = "ON" if device.is_on else "OFF"
                
            print(f"API Key: '{name}'")
            print(f"  -> UI Label: {device.label}")
            print(f"  -> Current State: {state_val}")
            print(f"  -> Object Type: {type(device).__name__}")
            print("-" * 45)

if __name__ == "__main__":
    asyncio.run(discover_devices())