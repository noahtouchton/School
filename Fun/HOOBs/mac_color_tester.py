import sys
import asyncio

try:
    from iaqualink.client import AqualinkClient
except ImportError:
    from iaqualink.system import AqualinkSystem as AqualinkClient

USERNAME = "cwtouch@bellsouth.net"
PASSWORD = "zynfuhXogzop6qiwqu!"

async def test_color(hue_value):
    try:
        hue = float(hue_value) % 360
    except ValueError:
        print("❌ Error: Provide a hue 0-360.")
        return

    # Updated mapping based on your board's 'supported_effects'
    if hue < 30: effect = "Magenta" 
    elif hue < 90: effect = "Alpine White"
    elif hue < 150: effect = "Spring Green"
    elif hue < 210: effect = "Sky Blue"
    elif hue < 260: effect = "Cobalt Blue"
    elif hue < 310: effect = "Violet"
    else: effect = "Emerald Rose"

    print(f"\n🎯 INPUT HUE: {hue}° -> 🛠️  JANDY EFFECT: {effect}")
    print("🔌 Connecting...")

    async with AqualinkClient(USERNAME, PASSWORD) as client:
        systems = await client.get_systems()
        system = list(systems.values())[0]
        devices = await system.get_devices()
        light = devices.get("aux_2")

        print(f"🚀 Sending 'set_effect_by_name({effect})'...")
        try:
            # THIS IS THE FIXED METHOD CALL
            await light.set_effect_by_name(effect)
            print("✅ SUCCESS! The light is sequencing.")
        except Exception as e:
            print(f"❌ Failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 mac_color_final.py <0-360>")
    else:
        asyncio.run(test_color(sys.argv[1]))