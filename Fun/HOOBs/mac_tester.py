import asyncio

try:
    from iaqualink.client import AqualinkClient
except ImportError:
    try:
        from iaqualink import AqualinkClient
    except ImportError:
        from iaqualink.system import AqualinkSystem as AqualinkClient

USERNAME = "cwtouch@bellsouth.net"
PASSWORD = "zynfuhXogzop6qiwqu!"  # <-- Put your real password here

async def mac_control_panel():
    print("🔌 Connecting to Jandy Cloud...")
    
    async with AqualinkClient(USERNAME, PASSWORD) as client:
        systems = await client.get_systems()
        if not systems:
            print("❌ Error: Could not connect to system.")
            return
        
        system = list(systems.values())[0]
        devices = await system.get_devices()
        
        print("✅ Connected! System ready.")
        
        while True:
            print("\n" + "="*30)
            print(" 🎛️  MAC TEST CONTROL PANEL  🎛️")
            print("="*30)
            print("1. 📊 Get Current Statuses")
            print("2. 💨 Turn Blower ON")
            print("3. 💨 Turn Blower OFF")
            print("4. 💡 Turn Spa Light ON")
            print("5. 💡 Turn Spa Light OFF")
            print("6. 🌊 Turn Pool Pump ON")
            print("7. 🌊 Turn Pool Pump OFF")
            print("0. 🚪 Exit")
            print("="*30)
            
            choice = input("Select an option (0-7): ")
            
            if choice == "1":
                print("\n--- CURRENT STATE ---")
                print(f"Blower (aux_1): {'ON' if devices['aux_1'].is_on else 'OFF'}")
                print(f"Light  (aux_2): {'ON' if devices['aux_2'].is_on else 'OFF'}")
                print(f"Pump   (pool_pump): {'ON' if devices['pool_pump'].is_on else 'OFF'}")
                print(f"Temp 1 (pool_heater): {'ON' if devices['pool_heater'].is_on else 'OFF'}")
                print(f"Temp 2 (spa_heater): {'ON' if devices['spa_heater'].is_on else 'OFF'}")
            
            elif choice == "2":
                print("Sending command...")
                await devices["aux_1"].turn_on()
                print("✅ Blower commanded ON.")
            elif choice == "3":
                print("Sending command...")
                await devices["aux_1"].turn_off()
                print("✅ Blower commanded OFF.")
            
            elif choice == "4":
                print("Sending command...")
                await devices["aux_2"].turn_on()
                print("✅ Spa Light commanded ON.")
            elif choice == "5":
                print("Sending command...")
                await devices["aux_2"].turn_off()
                print("✅ Spa Light commanded OFF.")
                
            elif choice == "6":
                print("Sending command...")
                await devices["pool_pump"].turn_on()
                print("✅ Pool Pump commanded ON.")
            elif choice == "7":
                print("Sending command...")
                await devices["pool_pump"].turn_off()
                print("✅ Pool Pump commanded OFF.")
                
            elif choice == "0":
                print("Closing connection...")
                break
            else:
                print("❌ Invalid choice.")

if __name__ == "__main__":
    asyncio.run(mac_control_panel())