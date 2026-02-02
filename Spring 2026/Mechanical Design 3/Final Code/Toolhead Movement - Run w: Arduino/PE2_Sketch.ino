#include "HX711.h"
#include <Servo.h>
#include <AccelStepper.h>

HX711 scale1; // Load cell near front of hopper (gears)
HX711 scale2; // Load cell near back of hopper

// HX711 Load Cell Setup
const uint8_t LOADCELL1_DOUT_PIN = 8;
const uint8_t LOADCELL1_SCK_PIN = 7;
const uint8_t LOADCELL2_DOUT_PIN = 4;
const uint8_t LOADCELL2_SCK_PIN = 3;
const int LOADCELL_CALIBRATION = 63.5;  // Calibration number for load cells
const uint8_t STOPPING_WEIGHT = 1000; // grams of up force before stopping

// Create servo objects
Servo servo1;  // Servo on pin 5 180-0 flips down
Servo servo2;  // Servo on pin 6 0-180 opens

// Define pin numbers
const uint8_t FLIP_SERVO_PIN = 5;
const uint8_t SCOOP_SERVO_PIN = 6;

// Define stepper motor
const uint8_t MOTOR_STEP_PIN = 10;
const uint8_t MOTOR_DIR_PIN = 9;
const int MAX_STEPS = 1400; // 100mm of travel w/pitch 1mm and 200 steps/rev, 1400 ish
const uint8_t MOTOR_DIR = 1;  // Sign convention is - speed is down in code. This flips it
AccelStepper stepper(AccelStepper::DRIVER, MOTOR_STEP_PIN, MOTOR_DIR_PIN);

// Define buttons to be used for testing and limiting stepper movement
const uint8_t BUTTON_PIN = 12;     // the number of the pushbutton pin
const uint8_t LIMIT_SWITCH_PIN = 2;     // the limit switch pin
const bool LIMIT_ACTIVE_LOW = true;

// global variables:
float fullWeight = 0;  // Stores the current weight registered by the load cells
float emptyWeight = 0;
int sampleNum = 1;
double percentDeposited = 0;


void setup() {
  // Initialize serial communication for debugging
  Serial.begin(9600);
  
  // Attach Load Cells
  scale1.begin(LOADCELL1_DOUT_PIN, LOADCELL1_SCK_PIN);
  scale1.set_scale(LOADCELL_CALIBRATION);
  scale2.begin(LOADCELL2_DOUT_PIN, LOADCELL2_SCK_PIN);
  scale2.set_scale(LOADCELL_CALIBRATION);

  // Attach Stepper Motor
  stepper.setMaxSpeed(1000);
  stepper.setAcceleration(500);

  // Attach buttons
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LIMIT_SWITCH_PIN, LIMIT_ACTIVE_LOW ? INPUT_PULLUP : INPUT);

  // Attach Servos
  servo1.attach(FLIP_SERVO_PIN, 500, 2500);  // Extended range: min 500μs, max 2500μs
  servo2.attach(SCOOP_SERVO_PIN, 500, 2500);  // Extended range: min 500μs, max 2500μs
  
  // Initiate homing protocol
  Serial.println("Homing All Motors...");
  resetAll();
  
  // Wait for user feedback to start testing
  Serial.println("Homing Complete. Confirm by pressing button...");
  while (digitalRead(BUTTON_PIN) == HIGH){}
  Serial.println("Starting Test...\n");
  delay(1000);
}

// Checks the current state of the limit switch
bool isLimitSwitchPressed(uint8_t pin) {
  const uint8_t state = digitalRead(pin);
  return LIMIT_ACTIVE_LOW ? (state == LOW) : (state == HIGH);
}

//
void zeroActuator() {
  // First, move away from limit switch if it's already pressed
  stepper.setCurrentPosition(0);
  stepper.setSpeed(500 * MOTOR_DIR); // Move in reverse direction
  unsigned long timeout = millis() + 5000; // 5 second timeout
  while (isLimitSwitchPressed(LIMIT_SWITCH_PIN) && millis() < timeout) {
    stepper.runSpeed();
  }
  stepper.setSpeed(0);
  stepper.stop(); // Ensure motor stops
  delay(100); // Brief pause
  
  // Move forward until limit switch is pressed
  stepper.setSpeed(-400 * MOTOR_DIR); // Set the speed (positive for one direction)
  timeout = millis() + 60000; // 60 second timeout for safety
  while (!isLimitSwitchPressed(LIMIT_SWITCH_PIN) && millis() < timeout) {
    stepper.runSpeed(); // Run at constant speed
  }
  
  // Stop the motor immediately when limit switch is hit
  stepper.setSpeed(0);
  stepper.stop(); // Ensure motor stops
  stepper.setCurrentPosition(0); // Zero position
  return;
}

void resetAll() {
  servo1.write(180);
  servo2.write(0);
  delay(2000);
  zeroActuator();
  delay(1000);
  scale1.tare();
  scale2.tare();
  return;
}

void loop() {
  // Begin testing 
  Serial.println("Sample " + (String) sampleNum + " Started");
  Serial.println("Please Insert Soil Bag");
  Serial.println("Press Button When Ready to Start...");
  while (digitalRead(BUTTON_PIN) == HIGH){}
  Serial.println("Preparing to Scoop...");

  // Reset all motors and scales
  resetAll();

  // Flip hopper and open scoop in same motion
  for (int pos = 0; pos <= 180; pos++) {
    servo1.write(180 - pos);
    servo2.write(pos);
    delay(1);
  }
  delay(2000);
  
  // Drive stepper motor downward until the one load cell experiences 50g or max steps hit
  stepper.setSpeed(500 * MOTOR_DIR);
  while (stepper.currentPosition() < MAX_STEPS) { // && (abs((float)(scale1.get_units(10) + scale2.get_units(10))/10) < STOPPING_WEIGHT)
    stepper.setSpeed(1000 * MOTOR_DIR);
    stepper.run();
  }

  stepper.stop(); // Ensure motor stops
  delay(3000);
  Serial.println("Scooping Soil...\n");
  // Scoop the soil, hopefully
  for (int pos = 0; pos <= 180; pos++) {
    servo2.write(180 - pos);
    delay(3);
  }

  // Moves hopper out of bag and flips it over
  zeroActuator();
  delay(1000);
  for (int pos = 0; pos <= 180; pos++) {
    servo1.write(pos);
    delay(5);
  }
  delay(5000);

  // Output the currently measured weight of the soil
  Serial.println("Scoop " + (String) sampleNum + " Completed!");
  servo1.detach();
  servo2.detach();
  fullWeight = (float)(scale1.get_units(100) + scale2.get_units(100))/10;
  servo1.attach(FLIP_SERVO_PIN, 500, 2500);  // Extended range: min 500μs, max 2500μs
  servo2.attach(SCOOP_SERVO_PIN, 500, 2500);  // Extended range: min 500μs, max 2500μs
  servo1.write(180);
  servo2.write(0);
  Serial.println("Weight of Soil: " + (String) fullWeight + "g\n");

  // Inform user to swap in vial
  Serial.println("Please Remove Bag and Insert Soil Testing Vial (remember to weigh it)");
  Serial.println("Press Button When Ready to Proceed...");
  while (digitalRead(BUTTON_PIN) == HIGH){}
  Serial.println("Depositing Soil...\n");

  // Deposit the soil, hopefully
  stepper.moveTo(1000);
  while(stepper.run()) {};
  stepper.stop();

  for (int pos = 0; pos <= 180; pos++) {
    servo2.write(pos);
    delay(5);
  }
  delay(5000);
  for (int pos = 0; pos <= 180; pos++) {
    servo2.write(180 - pos);
    delay(2);
  }
  delay(5000);

  // Measures the leftover dirt
  servo1.detach();
  servo2.detach();
  emptyWeight = (float)(scale1.get_units(100) + scale2.get_units(100))/10;
  servo1.attach(FLIP_SERVO_PIN, 500, 2500);  // Extended range: min 500μs, max 2500μs
  servo2.attach(SCOOP_SERVO_PIN, 500, 2500);  // Extended range: min 500μs, max 2500μs
  servo1.write(180);
  servo2.write(0);
  percentDeposited = (1 - ((double) emptyWeight / (double) fullWeight)) * 100;
  Serial.println("Soil Leftover: " + (String) emptyWeight + "g");
  Serial.println((String) percentDeposited + "% of Soil Deposited"); // If over 100% went negative
  delay(2000);
  Serial.println("Sample " + (String) sampleNum + " Complete\n");
  sampleNum++;
  zeroActuator();
}