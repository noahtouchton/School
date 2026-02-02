/*
  CoreXY + Full Homing (Xmin→Xmax, Ymin→Ymax) + Park + Soft Limits
  Arduino Mega 2560 + 2x DRV8825 (STEP/DIR)

  Motor A: STEP=4, DIR=5
  Motor B: STEP=2, DIR=3

  Endstops (INPUT_PULLUP, active LOW):
    X_MIN=22, X_MAX=23, Y_MIN=24, Y_MAX=25
*/

#include <AccelStepper.h>

// -------- Motor pins --------
#define A_STEP_PIN  4
#define A_DIR_PIN   5
#define B_STEP_PIN  2
#define B_DIR_PIN   3
#define A_EN_PIN   -1
#define B_EN_PIN   -1

// -------- Endstop pins --------
#define X_MIN_PIN 22
#define X_MAX_PIN 23
#define Y_MIN_PIN 24
#define Y_MAX_PIN 25
const bool ENDSTOP_ACTIVE_LOW = true; // pressed == LOW with INPUT_PULLUP

// -------- Motion tuning --------
const float MAX_SPEED_RUN   = 800.0;  // steps/s
const float ACCEL_RUN       = 400.0;  // steps/s^2
const float MAX_SPEED_HOMEF = 600.0;  // fast homing seek
const float ACCEL_HOMEF     = 600.0;
const float MAX_SPEED_HOMES = 200.0;  // slow homing seek
const float ACCEL_HOMES     = 400.0;
const int   STEP_PULSE_US   = 2;      // DRV8825 likes ~2 µs STEP high
const long  BIG_TRAVEL      = 1L << 29; // large seek distance

// -------- Measured travel (filled during homing) --------
long X_MAX_STEPS = 0;
long Y_MAX_STEPS = 0;

// Park/home inside the box (will be clamped after homing)
long HOME_X_STEPS = 1000;
long HOME_Y_STEPS = 1000;

// -------- Steppers --------
AccelStepper motorA(AccelStepper::DRIVER, A_STEP_PIN, A_DIR_PIN);
AccelStepper motorB(AccelStepper::DRIVER, B_STEP_PIN, B_DIR_PIN);

// -------- Utilities --------
inline void enableDriver(int enPin, bool on) {
  if (enPin == -1) return;
  pinMode(enPin, OUTPUT);
  // Most carriers: ENABLE low = enabled
  digitalWrite(enPin, on ? LOW : HIGH);
}

inline bool xMinTrig(){ int v=digitalRead(X_MIN_PIN); return ENDSTOP_ACTIVE_LOW ? (v==LOW) : (v==HIGH); }
inline bool xMaxTrig(){ int v=digitalRead(X_MAX_PIN); return ENDSTOP_ACTIVE_LOW ? (v==LOW) : (v==HIGH); }
inline bool yMinTrig(){ int v=digitalRead(Y_MIN_PIN); return ENDSTOP_ACTIVE_LOW ? (v==LOW) : (v==HIGH); }
inline bool yMaxTrig(){ int v=digitalRead(Y_MAX_PIN); return ENDSTOP_ACTIVE_LOW ? (v==LOW) : (v==HIGH); }

// CoreXY mapping
inline long corexy_A_from_XY(long X,long Y){ return X + Y; }
inline long corexy_B_from_XY(long X,long Y){ return X - Y; }
inline long currentX(){ return (motorA.currentPosition() + motorB.currentPosition()) / 2; }
inline long currentY(){ return (motorA.currentPosition() - motorB.currentPosition()) / 2; }

void moveToXY(long X, long Y){
  // clamp to known bounds (once discovered)
  if (X_MAX_STEPS > 0) { if (X < 0) X = 0; if (X > X_MAX_STEPS) X = X_MAX_STEPS; }
  if (Y_MAX_STEPS > 0) { if (Y < 0) Y = 0; if (Y > Y_MAX_STEPS) Y = Y_MAX_STEPS; }
  motorA.moveTo(corexy_A_from_XY(X, Y));
  motorB.moveTo(corexy_B_from_XY(X, Y));
}

void jogX(long d){ moveToXY(currentX()+d, currentY()); }
void jogY(long d){ moveToXY(currentX(),   currentY()+d); }

void setProfileRun(){
  motorA.setMaxSpeed(MAX_SPEED_RUN);   motorA.setAcceleration(ACCEL_RUN);
  motorB.setMaxSpeed(MAX_SPEED_RUN);   motorB.setAcceleration(ACCEL_RUN);
}
void setProfileFast(){
  motorA.setMaxSpeed(MAX_SPEED_HOMEF); motorA.setAcceleration(ACCEL_HOMEF);
  motorB.setMaxSpeed(MAX_SPEED_HOMEF); motorB.setAcceleration(ACCEL_HOMEF);
}
void setProfileSlow(){
  motorA.setMaxSpeed(MAX_SPEED_HOMES); motorA.setAcceleration(ACCEL_HOMES);
  motorB.setMaxSpeed(MAX_SPEED_HOMES); motorB.setAcceleration(ACCEL_HOMES);
}

// -------- Homing state machine --------
enum class HS {
  START,
  // X min
  X_FAST_TO_MIN, X_WAIT_MIN_FAST, X_BACKOFF_FROM_MIN, X_APPROACH_MIN_SLOW, X_ZERO_MIN,
  // X max
  X_FAST_TO_MAX, X_WAIT_MAX_FAST, X_BACKOFF_FROM_MAX, X_APPROACH_MAX_SLOW, X_RECORD_MAX,
  // Y min
  Y_FAST_TO_MIN, Y_WAIT_MIN_FAST, Y_BACKOFF_FROM_MIN, Y_APPROACH_MIN_SLOW, Y_ZERO_MIN,
  // Y max
  Y_FAST_TO_MAX, Y_WAIT_MAX_FAST, Y_BACKOFF_FROM_MAX, Y_APPROACH_MAX_SLOW, Y_RECORD_MAX,
  PARK, DONE
};

HS hs = HS::START;
bool needHome = true;

void serviceHoming(){
  switch (hs) {
    case HS::START:{
      motorA.setCurrentPosition(0);
      motorB.setCurrentPosition(0);
      motorA.setMinPulseWidth(STEP_PULSE_US);
      motorB.setMinPulseWidth(STEP_PULSE_US);
      setProfileFast();
      hs = HS::X_FAST_TO_MIN;
    } break;

    // ---- X → MIN ----
    case HS::X_FAST_TO_MIN:{
      moveToXY(currentX() - BIG_TRAVEL, currentY());
      hs = HS::X_WAIT_MIN_FAST;
    } break;
    case HS::X_WAIT_MIN_FAST:{
      if (xMinTrig()) {
      motorA.stop(); motorB.stop();
      // --- DEBUG: report switch hit and current positions ---
      Serial.print(F("[HOME] X_MIN HIT at A=")); Serial.print(motorA.currentPosition());
      Serial.print(F(" B="));                    Serial.print(motorB.currentPosition());
      Serial.print(F("  => X="));                Serial.println(currentX());

      setProfileFast();
      jogX(+1000);
      hs = HS::X_BACKOFF_FROM_MIN;
      }
    } break;

    case HS::X_BACKOFF_FROM_MIN:{
      if (motorA.distanceToGo()==0 && motorB.distanceToGo()==0) { setProfileSlow(); jogX(-1500); hs = HS::X_APPROACH_MIN_SLOW; }
    } break;
    case HS::X_APPROACH_MIN_SLOW:{
      if (xMinTrig()) { motorA.stop(); motorB.stop(); hs = HS::X_ZERO_MIN; }
    } break;
    case HS::X_ZERO_MIN:{
      long a_at_zero = corexy_A_from_XY(0, currentY());
      long b_at_zero = corexy_B_from_XY(0, currentY());
      motorA.setCurrentPosition(a_at_zero);
      motorB.setCurrentPosition(b_at_zero);
      Serial.print(F("[HOME] X ZEROED at switch. Now X="));
      Serial.println(currentX());
      setProfileFast();
      hs = HS::X_FAST_TO_MAX;
    } break;

    // ---- X → MAX ----
    case HS::X_FAST_TO_MAX:{
      moveToXY(currentX() + BIG_TRAVEL, currentY());
      hs = HS::X_WAIT_MAX_FAST;
    } break;
    case HS::X_WAIT_MAX_FAST:{
      if (xMaxTrig()) { motorA.stop(); motorB.stop(); setProfileFast(); jogX(-1000); hs = HS::X_BACKOFF_FROM_MAX; }
    } break;
    case HS::X_BACKOFF_FROM_MAX:{
      if (motorA.distanceToGo()==0 && motorB.distanceToGo()==0) { setProfileSlow(); jogX(+1500); hs = HS::X_APPROACH_MAX_SLOW; }
    } break;
    case HS::X_APPROACH_MAX_SLOW:{
      if (xMaxTrig()) { motorA.stop(); motorB.stop(); 
      Serial.print(F("[HOME] X_MAX HIT at A="));
      Serial.print(motorA.currentPosition());
      Serial.print(F(" B="));
      Serial.print(motorB.currentPosition());
      Serial.print(F(" => X="));
      Serial.println(currentX());
      hs = HS::X_RECORD_MAX; };
    } break;
    case HS::X_RECORD_MAX:{
      X_MAX_STEPS = currentX();
      Serial.print(F("[HOME] X SPAN measured: 0..."));
      Serial.print(X_MAX_STEPS);
      Serial.print(F(" width=")); 
      Serial.print(X_MAX_STEPS); 
      Serial.println(F(" steps"));
      HOME_X_STEPS = X_MAX_STEPS / 2;

      hs = HS::Y_FAST_TO_MIN;
    } break;

    // ---- Y → MIN ----
    case HS::Y_FAST_TO_MIN:{
      moveToXY(currentX(), currentY() - BIG_TRAVEL);
      hs = HS::Y_WAIT_MIN_FAST;
    } break;
    case HS::Y_WAIT_MIN_FAST:{
      if (yMinTrig()) { motorA.stop(); motorB.stop(); setProfileFast(); jogY(+1000); hs = HS::Y_BACKOFF_FROM_MIN; }
    } break;
    case HS::Y_BACKOFF_FROM_MIN:{
      if (motorA.distanceToGo()==0 && motorB.distanceToGo()==0) { setProfileSlow(); jogY(-1500); hs = HS::Y_APPROACH_MIN_SLOW; }
    } break;
    case HS::Y_APPROACH_MIN_SLOW:{
      if (yMinTrig()) { motorA.stop(); motorB.stop(); hs = HS::Y_ZERO_MIN; }
    } break;
    case HS::Y_ZERO_MIN:{
      long a_at_zero = corexy_A_from_XY(currentX(), 0);
      long b_at_zero = corexy_B_from_XY(currentX(), 0);
      motorA.setCurrentPosition(a_at_zero);
      motorB.setCurrentPosition(b_at_zero);
      setProfileFast();
      hs = HS::Y_FAST_TO_MAX;
    } break;

    // ---- Y → MAX ----
    case HS::Y_FAST_TO_MAX:{
      moveToXY(currentX(), currentY() + BIG_TRAVEL);
      hs = HS::Y_WAIT_MAX_FAST;
    } break;
    case HS::Y_WAIT_MAX_FAST:{
      if (yMaxTrig()) { motorA.stop(); motorB.stop(); setProfileFast(); jogY(-1000); hs = HS::Y_BACKOFF_FROM_MAX; }
    } break;
    case HS::Y_BACKOFF_FROM_MAX:{
      if (motorA.distanceToGo()==0 && motorB.distanceToGo()==0) { setProfileSlow(); jogY(+1500); hs = HS::Y_APPROACH_MAX_SLOW; }
    } break;
    case HS::Y_APPROACH_MAX_SLOW:{
      if (yMaxTrig()) { motorA.stop(); motorB.stop(); hs = HS::Y_RECORD_MAX; }
    } break;
    case HS::Y_RECORD_MAX:{
      Y_MAX_STEPS = currentY();
      hs = HS::PARK;
    } break;

    // ---- Park and finish ----
    case HS::PARK:{
      if (HOME_X_STEPS < 0) HOME_X_STEPS = 0;
      if (HOME_Y_STEPS < 0) HOME_Y_STEPS = 0;
      if (HOME_X_STEPS > X_MAX_STEPS) HOME_X_STEPS = X_MAX_STEPS;
      if (HOME_Y_STEPS > Y_MAX_STEPS) HOME_Y_STEPS = Y_MAX_STEPS;

      // Rebind current A/B counters to current XY (for cleanliness)
      long a_now = corexy_A_from_XY(currentX(), currentY());
      long b_now = corexy_B_from_XY(currentX(), currentY());
      motorA.setCurrentPosition(a_now);
      motorB.setCurrentPosition(b_now);

      setProfileRun();
      moveToXY(HOME_X_STEPS, HOME_Y_STEPS);
      hs = HS::DONE;
      needHome = false;

      Serial.print(F("[HOME] X:[0..")); Serial.print(X_MAX_STEPS);
      Serial.print(F("] Y:[0.."));       Serial.print(Y_MAX_STEPS);
      Serial.println(F("]"));
    } break;

    case HS::DONE: break;
  }
}

void setup(){
  Serial.begin(115200);

  pinMode(X_MIN_PIN, INPUT_PULLUP);
  pinMode(X_MAX_PIN, INPUT_PULLUP);
  pinMode(Y_MIN_PIN, INPUT_PULLUP);
  pinMode(Y_MAX_PIN, INPUT_PULLUP);

  enableDriver(A_EN_PIN, true);
  enableDriver(B_EN_PIN, true);

  motorA.setMinPulseWidth(STEP_PULSE_US);
  motorB.setMinPulseWidth(STEP_PULSE_US);
  setProfileRun();

  hs = HS::START;
  needHome = true;
}

void loop() {
  // Always advance steppers
  motorA.run();
  motorB.run();

  // ---- Quick helpers local to loop ----
  auto endstopRead = [](int pin, bool activeLow) -> bool {
    if (pin < 0) return false;                     // not wired
    int v = digitalRead(pin);
    return activeLow ? (v == LOW) : (v == HIGH);   // pressed?
  };
  const bool xMinHit = endstopRead(X_MIN_PIN, ENDSTOP_ACTIVE_LOW);
  const bool xMaxHit = endstopRead(X_MAX_PIN, ENDSTOP_ACTIVE_LOW);

  // Simple arrival check for both motors
  const bool arrived = (motorA.distanceToGo() == 0 && motorB.distanceToGo() == 0);

  // ---- X-only homing FSM (all inside loop via static state) ----
  enum State {
    START,              // arm homing
    SEEK_X_MIN_FAST,    // drive toward -X until X_MIN
    SET_X_ZERO,         // bind X=0 at the switch
    SEEK_X_MAX_FAST,    // (optional) drive toward +X until X_MAX
    SET_X_MAX,          // record X_MAX_STEPS
    PARK,               // move to HOME_X/HOME_Y (or X_MAX/2 fallback)
    READY               // done; normal motion can run
  };
  static State st = START;
  static bool homingNeeded = true;

  // Big seek distance for one-shot moves
  const long BIG = 1L << 29;

  switch (st) {
    case START: {
      if (!homingNeeded) { st = READY; break; }

      // Optional: ensure a sane pulse width for DRV8825
      motorA.setMinPulseWidth(2);
      motorB.setMinPulseWidth(2);

      // Use a moderate homing profile (you can tweak)
      motorA.setMaxSpeed(600); motorA.setAcceleration(600);
      motorB.setMaxSpeed(600); motorB.setAcceleration(600);

      // Begin seeking X_MIN
      moveToXY(currentX() - BIG, currentY());
      st = SEEK_X_MIN_FAST;
      Serial.println(F("[HOME] Seeking X_MIN..."));
    } break;

    case SEEK_X_MIN_FAST: {
      if (xMinHit) {
        motorA.stop(); motorB.stop();           // decel to stop
        st = SET_X_ZERO;
        Serial.println(F("[HOME] X_MIN hit -> set X=0"));
      }
      // If for some reason we arrived without hitting the switch (bench test),
      // extend the seek again.
      else if (arrived) {
        moveToXY(currentX() - BIG, currentY());
      }
    } break;

    case SET_X_ZERO: {
      // Bind virtual X=0 at current Y
      long a_at_zero = corexy_A_from_XY(0, currentY());
      long b_at_zero = corexy_B_from_XY(0, currentY());
      motorA.setCurrentPosition(a_at_zero);
      motorB.setCurrentPosition(b_at_zero);

      // If we have an X_MAX switch, go find it; otherwise skip to PARK
      if (X_MAX_PIN >= 0) {
        moveToXY(currentX() + BIG, currentY());
        st = SEEK_X_MAX_FAST;
        Serial.println(F("[HOME] Seeking X_MAX..."));
      } else {
        st = SET_X_MAX; // fall through to set a config/fallback value
      }
    } break;

    case SEEK_X_MAX_FAST: {
      if (xMaxHit) {
        motorA.stop(); motorB.stop();
        st = SET_X_MAX;
        Serial.println(F("[HOME] X_MAX hit -> record travel"));
      } else if (arrived) {
        moveToXY(currentX() + BIG, currentY()); // keep going until hit
      }
    } break;

    case SET_X_MAX: {
      // If you declared X_MAX_STEPS globally, record measured travel.
      // If not, or if no X_MAX switch, use a configured fallback.
      #ifdef X_MAX_STEPS
        if (X_MAX_PIN >= 0) {
          X_MAX_STEPS = currentX();             // measured
        } else {
          // If you have CONFIG_X_MAX_STEPS defined elsewhere, use it.
          // Otherwise use a conservative default.
          if (X_MAX_STEPS <= 0) X_MAX_STEPS = 50000;
        }
      #endif

      // Switch back to run profile (slower or faster as you like)
      motorA.setMaxSpeed(800); motorA.setAcceleration(400);
      motorB.setMaxSpeed(800); motorB.setAcceleration(400);

      st = PARK;

      // Status print (doesn't require X_MAX_STEPS to exist)
      Serial.print(F("[HOME] X range: 0.."));
      #ifdef X_MAX_STEPS
        Serial.println(X_MAX_STEPS);
      #else
        Serial.println(F("(set)"));
      #endif
    } break;

    case PARK: {
      // Choose park/“home” target.
      long targetX, targetY;

      #ifdef HOME_X_STEPS
        targetX = HOME_X_STEPS;
      #else
        #ifdef X_MAX_STEPS
          targetX = (X_MAX_STEPS > 0) ? (X_MAX_STEPS / 2) : 1000;  // center or 1000
        #else
          targetX = 1000;  // fallback if not defined
        #endif
      #endif

      #ifdef HOME_Y_STEPS
        targetY = HOME_Y_STEPS;
      #else
        targetY = 0;      // we didn't home Y; leave it where you like
      #endif

      // Clamp inside [0..X_MAX] if available
      #ifdef X_MAX_STEPS
        if (targetX < 0) targetX = 0;
        if (X_MAX_STEPS > 0 && targetX > X_MAX_STEPS) targetX = X_MAX_STEPS;
      #endif

      moveToXY(targetX, targetY);
      st = READY;
      homingNeeded = false;

      Serial.print(F("[HOME] Parking at X=")); Serial.print(targetX);
      Serial.print(F(" Y=")); Serial.println(targetY);
    } break;

    case READY: {
      // Homing finished. Do whatever you want here.
      // Example: simple X ping-pong within known range (if defined)
      static bool dir = true;
      if (arrived) {
        long xLo = 0;
        long xHi =
        #ifdef X_MAX_STEPS
          (X_MAX_STEPS > 0 ? X_MAX_STEPS : 2000);
        #else
          2000;
        #endif
        moveToXY(dir ? xHi : xLo, currentY());
        dir = !dir;
      }
    } break;
  }
}

