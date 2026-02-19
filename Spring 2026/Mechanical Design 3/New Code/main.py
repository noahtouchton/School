import sys
import signal
from PySide6.QtCore import QCoreApplication, QTimer
from arduino_worker import ArduinoWorker
import time

DIST = 5.0 #inches between soil bags
ROWS = 3
COLS = 3


class GantryController:
    def __init__(self, gan_worker, scoop_worker, stir_worker, rows, cols):
        self.gan_worker = gan_worker
        self.scoop_worker = scoop_worker
        self.stir_worker = stir_worker
        self.rows = rows
        self.cols = cols
        self.current_row = 0
        self.current_col = 0
        self.soil_types = []

        self.gan_worker.ack.connect(self.start_stir) #when the arduino receives finished move to the next command
        self.stir_worker.ack.connect(self.start_scoop) #when the arduino receives finished move to the next command
        #add the camera stuff here once implemented
    
        self.scoop_worker.ack.connect(self.handle_scoop_ack) #when the arduino receives finished

    def start_process(self):
        self.gan_worker.start()
        self.stir_worker.start()
        self.scoop_worker.start()
        QTimer.singleShot(2000, self.send_gantry_move) #start after 2 seconds

    def send_gantry_move(self):
        if self.current_row < self.rows:
            x = self.current_col * DIST
            y = self.current_row * DIST
            print(f"Moving to bag at row {self.current_row}, col {self.current_col} (x={x}, y={y})")
            self.gan_worker.send_raw(f"{x},{y}")
        else:
            print("All bags processed!")

    def start_stir(self, last_cmd):
        print("Starting stir process...")
        self.stir_worker.send_raw("1") 

    def start_scoop(self, last_cmd):
        print("Starting scoop process...")
        self.scoop_worker.send_raw("1") 

    def start_unscoop(self):
        print("Starting unscoop process...")
        self.scoop_worker.send_raw("0")
    
    def handle_scoop_ack(self, last_cmd):
        if last_cmd == "1":
            print("Scoop finished. Taking picture...")
            self.capture_image()
        elif last_cmd == "0":
            print("Unscoop finished. Moving to next bag...")
            self.finish_bag_cycle(last_cmd)
    
    def capture_image(self):
        #simulate camera delay
        soil_type = 1 #placeholder for actual image processing result
        self.soil_types.append(soil_type)
        QTimer.singleShot(1000, self.start_unscoop)

    def finish_bag_cycle(self, last_cmd):
        print("Finished bag cycle. Calculating next move...")
        self.current_col += 1
        if self.current_col >= self.cols:
            self.current_col = 0
            self.current_row += 1
        self.send_gantry_move()

    def display_results(self):
        print("All soil types:", self.soil_types)
# Helper to close the app cleanly with Ctrl+C
def signal_handler(sig, frame):
    print("Exiting...")
    app.quit()

if __name__ == "__main__":
    # Use QCoreApplication for non-GUI (headless) or QApplication for GUI
    app = QCoreApplication(sys.argv)
    
    # Allow Ctrl+C to kill the app
    signal.signal(signal.SIGINT, signal_handler)
    timer = QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None) 

    # --- SETUP ARDUINOS ---
    
    # ARDUINO 1: Identify by Serial Number (Recommended)
    # Run 'python -m serial.tools.list_ports -v' in terminal to find this number.
    ard_gan = ArduinoWorker(target_serial_number="48CA435A3A20")
    ard_stir = ArduinoWorker(target_serial_number="95138323838351401091", baud=9600)
    ard_scoop = ArduinoWorker(target_serial_number="44231313430351116261", baud=9600)

    #CRITICAL: Connect signals so you see what's happening!
    ard_gan.log.connect(lambda s: print(f"[LOG]: {s}"))
    ard_gan.ack.connect(lambda s: print(f"[ACK]: {s}"))
    ard_gan.error.connect(lambda s: print(f"[ERR]: {s}"))
    ard_gan.connected.connect(lambda s: print(f"[SYS]: Connected to {s}"))

    ard_stir.log.connect(lambda s: print(f"[LOG]: {s}"))
    ard_stir.ack.connect(lambda s: print(f"[ACK]: {s}"))
    ard_stir.error.connect(lambda s: print(f"[ERR]: {s}"))
    ard_stir.connected.connect(lambda s: print(f"[SYS]: Connected to {s}"))

    ard_scoop.log.connect(lambda s: print(f"[LOG]: {s}"))
    ard_scoop.ack.connect(lambda s: print(f"[ACK]: {s}"))
    ard_scoop.error.connect(lambda s: print(f"[ERR]: {s}"))
    ard_scoop.connected.connect(lambda s: print(f"[SYS]: Connected to {s}"))

    controller = GantryController(ard_gan, ard_scoop, ard_stir, ROWS, COLS)
    controller.start_process()

    controller.display_results()

    # Start the Event Loop
    sys.exit(app.exec())

  