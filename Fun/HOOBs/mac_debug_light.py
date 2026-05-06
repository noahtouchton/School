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
PASSWORD = "zynfuhXogzop6qiwqu!"

async def debug_light():
    print("🔌 Connecting to Jandy Cloud...")
    async with AqualinkClient(USERNAME, PASSWORD) as client:
        systems = await client.get_systems()
        if not systems:
            print("❌ Error: Could not connect to Jandy cloud.")
            return
        
        system = list(systems.values())[0]
        devices = await system.get_devices()
        light = devices.get("aux_2")
        
        print("\n" + "="*45)
        print(f"🔦 OBJECT TYPE: {type(light)}")
        print(f"🧬 BASE CLASSES: {type(light).__bases__}")
        print("\n🛠️  AVAILABLE METHODS & ATTRIBUTES:")
        
        # Get all non-private attributes
        attributes = [attr for attr in dir(light) if not attr.startswith('_')]
        for attr in attributes:
            try:
                val = getattr(light, attr)
                if callable(val):
                    print(f"  -> {attr}()  [METHOD]")
                else:
                    print(f"  -> {attr} = {val}  [PROPERTY]")
            except Exception as e:
                print(f"  -> {attr} (Error reading: {e})")
                
        print("="*45 + "\n")

if __name__ == "__main__":
    asyncio.run(debug_light())