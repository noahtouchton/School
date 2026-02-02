// ---------------------- Motor Pin Assignments ----------------------
// Motor A drives the belt segment that participates in X + Y motion.
const uint8_t MOTOR_A_STEP_PIN = 4;
const uint8_t MOTOR_A_DIR_PIN  = 5;

// Motor B drives the belt segment that participates in X - Y motion.
const uint8_t MOTOR_B_STEP_PIN = 2;
const uint8_t MOTOR_B_DIR_PIN  = 3;

// Enable pins are optional.  Leave at -1 if your drivers are always enabled.
const int MOTOR_A_ENABLE_PIN = -1;
const int MOTOR_B_ENABLE_PIN = -1;

// ---------------------- Limit Switch Assignments ----------------------
// Limit switches sit at the minimum (home) end of each axis.
const uint8_t X_MIN_LIMIT_PIN = 23;  // Long side of the bed (X axis)
const uint8_t Y_MIN_LIMIT_PIN = 22;  // Short side of the bed (Y axis)

// Maximum limit switches sit at the far end of each axis.
const uint8_t X_MAX_LIMIT_PIN = 25;  // Long side max (X axis)
const uint8_t Y_MAX_LIMIT_PIN = 24;  // Short side max (Y axis)

// Set to true for switches that pull the pin LOW when pressed (recommended).
const bool LIMIT_ACTIVE_LOW = true;

// ---------------------- Motion Tuning ----------------------
// Travel motion (general use after homing)
const float TRAVEL_MAX_SPEED = 1200.0f;  // steps per second
const float TRAVEL_ACCEL     = 800.0f;   // steps per second^2

// Homing motion
const float HOMING_SPEED     = 1500.0f;   // approach speed toward the switch
const float HOMING_ACCEL     = 800.0f;   // acceleration during homing
const float BACKOFF_SPEED    = 300.0f;   // speed when backing away
const float BACKOFF_ACCEL    = 600.0f;   // acceleration when backing away
const long  BACKOFF_STEPS    = 200L;     // steps to retreat after triggering
#include <AccelStepper.h>
// ---------------------- Stepper Objects ----------------------
AccelStepper motorA(AccelStepper::DRIVER, MOTOR_A_STEP_PIN, MOTOR_A_DIR_PIN);
AccelStepper motorB(AccelStepper::DRIVER, MOTOR_B_STEP_PIN, MOTOR_B_DIR_PIN);

// ---------------------- Calibration Variables ----------------------
long maxXSteps = 0L;  // Will be set during calibration
long maxYSteps = 0L;  // Will be set during calibration

// ---------------------- CoreXY Helper Functions ----------------------
inline long coreXY_A(long xSteps, long ySteps) { return xSteps + ySteps; }
inline long coreXY_B(long xSteps, long ySteps) { return xSteps - ySteps; }
inline long currentX() {
  return (motorA.currentPosition() + motorB.currentPosition()) / 2;
}
inline long currentY() {
  return (motorA.currentPosition() - motorB.currentPosition()) / 2;
}
inline float absf(float value) { return (value >= 0.0f) ? value : -value; }

// ---------------------- Utility Prototypes ----------------------
void enableDriverPin(int pin, bool enableActiveLow);
bool isLimitPressed(uint8_t pin);
bool checkMaxLimits();
void setXYPosition(long xSteps, long ySteps);
void moveCoreXYRelative(long deltaX, long deltaY, float maxSpeed, float accel, bool checkLimits = true);
void homeAxisX();
void homeAxisY();
void homeSystem();
void calibrateMaxX();
void calibrateMaxY();
void calibrateMaxLimits();

// ---------------------- Setup ----------------------
void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial monitor on boards that need it (harmless on Mega)
  }
  Serial.println();
  Serial.println(F("Desktop Soil Sampler – CoreXY Calibration"));
  Serial.println(F("Initializing..."));

  // Configure limit switches with internal pull-ups if they are active LOW.
  pinMode(X_MIN_LIMIT_PIN, LIMIT_ACTIVE_LOW ? INPUT_PULLUP : INPUT);
  pinMode(Y_MIN_LIMIT_PIN, LIMIT_ACTIVE_LOW ? INPUT_PULLUP : INPUT);
  pinMode(X_MAX_LIMIT_PIN, LIMIT_ACTIVE_LOW ? INPUT_PULLUP : INPUT);
  pinMode(Y_MAX_LIMIT_PIN, LIMIT_ACTIVE_LOW ? INPUT_PULLUP : INPUT);

  // Configure (optional) enable pins.
  enableDriverPin(MOTOR_A_ENABLE_PIN, true);
  enableDriverPin(MOTOR_B_ENABLE_PIN, true);

  // Configure steppers for general travel.
  motorA.setMaxSpeed(TRAVEL_MAX_SPEED);
  motorA.setAcceleration(TRAVEL_ACCEL);
  motorB.setMaxSpeed(TRAVEL_MAX_SPEED);
  motorB.setAcceleration(TRAVEL_ACCEL);

  Serial.println(F("Starting homing sequence..."));
  homeSystem();
  Serial.println(F("Homing complete. System ready at (0, 0)."));
  
  Serial.println(F("Starting calibration of max limits..."));
  calibrateMaxLimits();
  Serial.println(F("Calibration complete."));
}

// ---------------------- Main Loop ----------------------
void loop() {
  // Idle placeholder.  Insert motion commands here after calibration.
  delay(100);
}

// ---------------------- Utility Implementations ----------------------
void enableDriverPin(int pin, bool enableActiveLow) {
  if (pin < 0) {
    return; // Not used
  }
  pinMode(pin, OUTPUT);
  // Enable the driver (most stepper drivers are active low on ENABLE).
  digitalWrite(pin, enableActiveLow ? LOW : HIGH);
}

bool isLimitPressed(uint8_t pin) {
  const int state = digitalRead(pin);
  return LIMIT_ACTIVE_LOW ? (state == LOW) : (state == HIGH);
}

bool checkMaxLimits() {
  // Returns true if any max limit switch is triggered
  if (isLimitPressed(X_MAX_LIMIT_PIN)) {
    Serial.println(F("WARNING: X MAX limit triggered!"));
    return true;
  }
  if (isLimitPressed(Y_MAX_LIMIT_PIN)) {
    Serial.println(F("WARNING: Y MAX limit triggered!"));
    return true;
  }
  return false;
}

void setXYPosition(long xSteps, long ySteps) {
  motorA.setCurrentPosition(coreXY_A(xSteps, ySteps));
  motorB.setCurrentPosition(coreXY_B(xSteps, ySteps));
}

void moveCoreXYRelative(long deltaX, long deltaY, float maxSpeed, float accel, bool checkLimits) {
  const long targetX = currentX() + deltaX;
  const long targetY = currentY() + deltaY;

  const long targetA = coreXY_A(targetX, targetY);
  const long targetB = coreXY_B(targetX, targetY);

  motorA.setAcceleration(accel);
  motorB.setAcceleration(accel);
  motorA.setMaxSpeed(absf(maxSpeed));
  motorB.setMaxSpeed(absf(maxSpeed));

  motorA.moveTo(targetA);
  motorB.moveTo(targetB);

  while (motorA.distanceToGo() != 0L || motorB.distanceToGo() != 0L) {
    // Check max limits during motion and abort if triggered (only if checkLimits is true)
    if (checkLimits && checkMaxLimits()) {
      motorA.stop();
      motorB.stop();
      Serial.println(F("Motion aborted due to max limit switch!"));
      break;
    }
    motorA.run();
    motorB.run();
  }
}

void homeAxisX() {
  Serial.println(F(" -> Homing X axis (long side)"));

  if (isLimitPressed(X_MIN_LIMIT_PIN)) {
    Serial.println(F("    X limit already active. Backing off before homing."));
    moveCoreXYRelative(0L, BACKOFF_STEPS,BACKOFF_SPEED, BACKOFF_ACCEL);
  }

  //const float speedX = -HOMING_SPEED; // Move toward the minimum endstop
 // const float speedY = 0.0f;
 // const float speedA = speedX + speedY;
  //Const float speedB = speedX - speedY;

  motorA.setAcceleration(HOMING_ACCEL);
  motorB.setAcceleration(HOMING_ACCEL);
  motorA.setMaxSpeed(absf(HOMING_SPEED));
  motorB.setMaxSpeed(absf(HOMING_SPEED));
  motorA.setSpeed(HOMING_SPEED);
  motorB.setSpeed(-HOMING_SPEED);

  Serial.println(F("    Seeking X minimum limit..."));
  while (!isLimitPressed(X_MIN_LIMIT_PIN)) {
    motorA.runSpeed();
    motorB.runSpeed();
  }
  motorA.setSpeed(0.0f);
  motorB.setSpeed(0.0f);
  Serial.println(F("    X limit triggered."));
  delay(50); // Allow the switch to settle.

  Serial.println(F("    Backing off X axis to clear the switch."));
  moveCoreXYRelative(0L, -BACKOFF_STEPS, BACKOFF_SPEED, BACKOFF_ACCEL);
  setXYPosition(0L, currentY());
  Serial.println(F("    X axis zero set."));
}

void homeAxisY() {
  Serial.println(F(" -> Homing Y axis (short side)"));

  if (isLimitPressed(Y_MIN_LIMIT_PIN)) {
    Serial.println(F("    Y limit already active. Backing off before homing."));
    moveCoreXYRelative(0L, -BACKOFF_STEPS, BACKOFF_SPEED, BACKOFF_ACCEL);
  }

 // const float speedX = 0.0f;
  //const float speedY = HOMING_SPEED; // Move toward the minimum endstop.
  //const float speedA = speedX + speedY;
  //const float speedB = speedX - speedY;

  motorA.setAcceleration(HOMING_ACCEL);
  motorB.setAcceleration(HOMING_ACCEL);
  motorA.setMaxSpeed(absf(HOMING_SPEED));
  motorB.setMaxSpeed(absf(HOMING_SPEED));
  motorA.setSpeed(-HOMING_SPEED);  // Negate to compensate for Motor A direction wiring: -(-HOMING_SPEED) = +HOMING_SPEED
  motorB.setSpeed(-HOMING_SPEED);  // Negate Motor B: -(+HOMING_SPEED) = -HOMING_SPEED (opposite direction from Motor A)

  Serial.println(F("    Seeking Y minimum limit..."));
  while (!isLimitPressed(Y_MIN_LIMIT_PIN)) {
    motorA.runSpeed();
    motorB.runSpeed();
  }
  motorA.setSpeed(0.0f);
  motorB.setSpeed(0.0f);
  Serial.println(F("    Y limit triggered."));
  delay(50); // Allow the switch to settle.

  Serial.println(F("    Backing off Y axis to clear the switch."));
  moveCoreXYRelative(BACKOFF_STEPS, 0L, BACKOFF_SPEED, BACKOFF_ACCEL);
  setXYPosition(currentX(), 0L);
  Serial.println(F("    Y axis zero set."));
}

void homeSystem() {
  homeAxisX();
  homeAxisY();
  Serial.print(F(" -> Homing finished. Current position: X="));
  Serial.print(currentX());
  Serial.print(F(" steps, Y="));
  Serial.print(currentY());
  Serial.println(F(" steps."));
}

void calibrateMaxX() {
  Serial.println(F(" -> Calibrating X axis maximum"));

  //const float speedX = HOMING_SPEED; // Move toward the maximum endstop (positive direction).
  //const float speedY = 0.0f;
  //const float speedA = speedX + speedY;
  //const float speedB = speedX - speedY;

  motorA.setAcceleration(HOMING_ACCEL);
  motorB.setAcceleration(HOMING_ACCEL);
  motorA.setMaxSpeed(absf(HOMING_SPEED));
  motorB.setMaxSpeed(absf(HOMING_SPEED));
  motorA.setSpeed(-HOMING_SPEED);
  motorB.setSpeed(HOMING_SPEED);

  Serial.println(F("    Seeking X maximum limit..."));
  while (!isLimitPressed(X_MAX_LIMIT_PIN)) {
    motorA.runSpeed();
    motorB.runSpeed();
  }
  motorA.setSpeed(0.0f);
  motorB.setSpeed(0.0f);
  Serial.println(F("    X MAX limit triggered."));
  delay(50); // Allow the switch to settle.

  // Record the maximum X position
  maxXSteps = currentX();
  Serial.print(F("    X maximum position: "));
  Serial.print(maxXSteps);
  Serial.println(F(" steps"));

  Serial.println(F("    Backing off X axis to clear the switch."));
  moveCoreXYRelative(0L, BACKOFF_STEPS, BACKOFF_SPEED, BACKOFF_ACCEL, false);
}

void calibrateMaxY() {
  Serial.println(F(" -> Calibrating Y axis maximum"));

 // const float speedX = 0.0f;
  //const float speedY = HOMING_SPEED; // Move toward the maximum endstop (positive direction).
  //const float speedA = speedX + speedY;
  //const float speedB = speedX - speedY;

  motorA.setAcceleration(HOMING_ACCEL);
  motorB.setAcceleration(HOMING_ACCEL);
  motorA.setMaxSpeed(absf(HOMING_SPEED));
  motorB.setMaxSpeed(absf(HOMING_SPEED));
  motorA.setSpeed(HOMING_SPEED);
  motorB.setSpeed(HOMING_SPEED);

  Serial.println(F("    Seeking Y maximum limit..."));
  while (!isLimitPressed(Y_MAX_LIMIT_PIN)) {
    motorA.runSpeed();
    motorB.runSpeed();
  }
  maxYSteps = currentY();
  motorA.setSpeed(0.0f);
  motorB.setSpeed(0.0f);
  Serial.println(F("    Y MAX limit triggered."));
  delay(50); // Allow the switch to settle.

  // Record the maximum Y position
  maxYSteps = currentY();
  Serial.print(F("    Y maximum position: "));
  Serial.print(maxYSteps);
  Serial.print(currentY());
  Serial.println(F(" steps"));

  Serial.println(F("    Backing off Y axis to clear the switch."));
  moveCoreXYRelative(-BACKOFF_STEPS, 0L, BACKOFF_SPEED, BACKOFF_ACCEL, false);
}

void calibrateMaxLimits() {
  calibrateMaxY();
  calibrateMaxX();
  
  Serial.println(F(" -> Calibration summary:"));
  Serial.print(F("    X travel range: 0 to "));
  Serial.print(maxXSteps);
  Serial.println(F(" steps"));
  Serial.print(F("    Y travel range: 0 to "));
  Serial.print(maxYSteps);
 // Serial.print(currentY());
  Serial.println(F(" steps"));
  Serial.println(F("motorA.currentPosition: "));
  Serial.println(motorA.currentPosition());
  Serial.println(F("motorB.currentPosition: "));
  Serial.println(motorB.currentPosition());
  
  // Return to home position
  Serial.println(F(" -> Returning to home position..."));
  Serial.print(F("    Current position before return: X="));
  Serial.print(currentX());
  Serial.print(F(", Y="));
  Serial.println(currentY());
  moveCoreXYRelative(0L, -currentY(), TRAVEL_MAX_SPEED, TRAVEL_ACCEL);
 // Serial.print(currentX());
  //Serial.print(currentY());
  moveCoreXYRelative(-currentX(), 0L, TRAVEL_MAX_SPEED, TRAVEL_ACCEL);
  Serial.print(F("    Position after return: X="));
  Serial.print(currentX());
  Serial.print(F(", Y="));
  Serial.println(currentY());
  Serial.println(F(" -> Gantry at home (0, 0). Ready for operation."));
}
