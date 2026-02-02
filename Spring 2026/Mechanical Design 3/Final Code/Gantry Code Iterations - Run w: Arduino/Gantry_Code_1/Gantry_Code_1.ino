/*
  Minimal AccelStepper "Make It Move"  + CoreXY mapper
  Arduino Mega 2560 R3 + 2x step/dir drivers (A4988 / DRV8825 / TMCxx)

  - Two motors wired as a CoreXY stage (A/B belts)
  - X/Y "virtual axes" mapped to A/B motor step targets
  - AccelStepper handles accel/decel; non-blocking run()

  Wiring: STEP/DIR from these pins to your drivers. ENABLE optional.
*/

#include <AccelStepper.h>

// ------------ Pin assignments (Mega 2560) ------------
// Motor A (one of the CoreXY belts)
#define A_STEP_PIN  4
#define A_DIR_PIN   5
#define A_EN_PIN    -1    // set to -1 if not wired

// Motor B (the other CoreXY belt) -- use different pins from A
#define B_STEP_PIN  2     // CHANGED from 4 -> 5
#define B_DIR_PIN   3     // CHANGED from 4 -> 6
#define B_EN_PIN    -1    // set to -1 if not wired

// ------------ Motion tuning ------------
const float MAX_SPEED = 500.0;    // steps/s (motor shaft steps, *after* any microstepping)
const float ACCEL     = 200.0;    // steps/s^2
const long  TRAVEL    = 1000;     // X/Y travel in *steps* (virtual axis units)

// Optional axis inversion if your stage moves the opposite way you expect.
// (You can also flip DIR wiring on a driver instead.)
const bool INVERT_X = false;
const bool INVERT_Y = false;

// ------------ Stepper objects ------------
AccelStepper motorA(AccelStepper::DRIVER, A_STEP_PIN, A_DIR_PIN);
AccelStepper motorB(AccelStepper::DRIVER, B_STEP_PIN, B_DIR_PIN);

// ------------ Helpers ------------
inline void enableDriver(int enPin, bool on) {
  if (enPin == -1) return;
  pinMode(enPin, OUTPUT);
  // Most carriers are ENABLE LOW: LOW = enabled
  digitalWrite(enPin, on ? LOW : HIGH);
}

// ---- CoreXY mapping ----
// A = X + Y
// B = X - Y
// (all in steps)
inline long corexy_A_from_XY(long X, long Y) { return X + Y; }
inline long corexy_B_from_XY(long X, long Y) { return X - Y; }

// Inverse mapping to query where we are in X/Y "virtual" space.
// X = (A + B)/2
// Y = (A - B)/2
inline long currentX() {
  return (motorA.currentPosition() + motorB.currentPosition()) / 2;
}
inline long currentY() {
  return (motorA.currentPosition() - motorB.currentPosition()) / 2;
}

// Command a target in X/Y virtual space (in steps).
void moveToXY(long X, long Y) {
  // Apply optional inversions
  if (INVERT_X) X = -X;
  if (INVERT_Y) Y = -Y;

  const long aTarget = corexy_A_from_XY(X, Y);
  const long bTarget = corexy_B_from_XY(X, Y);

  motorA.moveTo(aTarget);
  motorB.moveTo(bTarget);
}

void setup() {
  Serial.begin(115200);

  enableDriver(A_EN_PIN, true);
  enableDriver(B_EN_PIN, true);

  motorA.setMaxSpeed(MAX_SPEED);
  motorA.setAcceleration(ACCEL);
  motorB.setMaxSpeed(MAX_SPEED);
  motorB.setAcceleration(ACCEL);

  // Start at X = -TRAVEL, Y = 0 (ping-pong along X as a demo)
  moveToXY(-TRAVEL, 0);

  // Note: AccelStepper sets STEP/DIR pins to OUTPUT internally.
}

void loop() {
  // Advance both motors toward their targets (non-blocking)
  motorA.run();
  motorB.run();

  // When BOTH motors arrive, we reached the XY target.
  if (motorA.distanceToGo() == 0 && motorB.distanceToGo() == 0) {
    // Demo pattern 1: X-axis ping-pong between -TRAVEL and +TRAVEL at Y=0
    static bool goingPositive = true;
    long nextX = goingPositive ? +TRAVEL : -TRAVEL;
    long nextY = 0;

    moveToXY(nextX, nextY);
    goingPositive = !goingPositive;

    // Debug print current virtual position:
    Serial.print("X="); Serial.print(currentX());
    Serial.print("  Y="); Serial.println(currentY());
  }
}
