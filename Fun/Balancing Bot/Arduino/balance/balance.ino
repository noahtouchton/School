/*
  Balancing bot -- LQG controller (LQR + observer, no encoders)

  States   x = [theta, thetaDot, phiDot]
             theta    = chassis lean from vertical            (rad)
             thetaDot = lean rate                             (rad/s)
             phiDot   = wheel speed  <-- ESTIMATED, not measured

  Measured y = [theta, thetaDot]   (complementary filter + gyro)
  Input    u = V                   (motor volts, before PWM scaling)

  Sign convention -- must match Dynamics.pdf and lqr.py:
      +theta  -> chassis top tips toward -Z
      +phi    -> bot drives toward +Z
      +V      -> drives wheels in the +phi sense

  Why an observer: feedback on [theta, thetaDot] alone CANNOT stabilize this
  bot. Motor torque is internal, so it cannot change angular momentum about
  the contact point -- one closed-loop root always stays in the right half
  plane regardless of gains. Estimating phiDot closes that loop. See lqr.py.

  BRING-UP ORDER -- do not skip:
    1. OPEN_LOOP 1, motors unplugged. Verify theta sign and magnitude.
    2. OPEN_LOOP 1, wheels OFF THE GROUND. Verify the commanded PWM drives
       the way you would move to catch the lean.
    3. Measure PWM_DEADBAND under load.
    4. OPEN_LOOP 0. Hold the bot. Release slowly.

  All gains regenerate from Python/lqr.py -- do not hand-edit them.
*/

#include <Wire.h>

// ============================================================
// BRING-UP SWITCH
// ============================================================
#define OPEN_LOOP 1    // 1 = print only, motors never driven
#define CALIBRATE 0       // 1 = run six-position accel calibration, then halt
#define MOTOR_TEST 0      // 1 = ramp each motor to find deadband + direction

// ============================================================
// PINS  (match inertiaCalc.ino)
// ============================================================
const int MOTOR_L_PWM  = 5;
const int MOTOR_L_DIR1 = 3;
const int MOTOR_L_DIR2 = 4;
const int MOTOR_R_PWM  = 6;
const int MOTOR_R_DIR1 = 7;
const int MOTOR_R_DIR2 = 8;

// ============================================================
// IMU
// ============================================================
const int MPU_ADDR = 0x68;
const uint8_t REG_PWR_MGMT_1  = 0x6B;
const uint8_t REG_GYRO_CONFIG = 0x1B;
const uint8_t REG_ACCEL_XOUT  = 0x3B;

// Gyro at +-500 deg/s. The +-250 default saturates at only ~20 deg of lean
// (a falling pendulum of this length hits 367 deg/s by 30 deg), and a pinned
// gyro reading corrupts theta exactly when you most need it.
const uint8_t GYRO_FS_SEL = 0x08;   // 0x00=250  0x08=500  0x10=1000  0x18=2000
const float GYRO_LSB = 65.5f;       // must match: 131 / 65.5 / 32.8 / 16.4
const float ACCEL_LSB = 16384.0f;   // +-2 g default

// --- accel offset / scale, from CALIBRATE mode --------------------------
// Needed because |a| measured 0.855 lying on one side and 1.030 held
// upright. A magnitude that CHANGES with orientation means per-axis offset;
// a constant wrong magnitude would be scale, which atan2 cancels for free.
// Offsets do not cancel -- they bias theta differently at every angle.
// Run with CALIBRATE 1, paste the printed values here, set CALIBRATE 0.
// Measured. Z carries a -0.16 g offset (+Z up read 0.860, -Z up read -1.180)
// -- twice the MPU-6050 spec, but Z is always the worst axis on these parts.
// All three SCALES came back within 2% of 1.0, so the sensitivity is fine and
// this is a genuine MPU-6050, not a clone.
const float ACC_OFF_X =  0.04041f, ACC_OFF_Y = -0.02156f, ACC_OFF_Z = -0.15978f;
const float ACC_SCL_X =  0.99327f, ACC_SCL_Y =  1.00084f, ACC_SCL_Z =  1.01994f;

// --- axis mapping: VERIFY THESE ON YOUR BUILD (step 1) ---------------
// theta = atan2(fore-aft axis, up axis).
//
// Standing upright, atan2(accX, accZ) returned 88.5 deg -- which means accX
// was ~1 and accZ ~0, i.e. the chip's X axis is the one pointing UP. The
// earlier readings that looked like a 17 deg mounting offset were taken with
// the bot on its SIDE, where Z dominated and the wrong mapping happened to
// produce a believable number.
//
// So: X = chassis up, Z = chassis fore-aft, Y = along the axle (unused for
// pitch, which is also why gyroY stays the pitch rate).
#define ACC_FOREAFT   accZ
#define ACC_UP        accX
#define GYRO_PITCH    gyroY
const float THETA_SIGN = 1.0f;      // flip to -1 if theta reads backwards
const float GYRO_SIGN  = 1.0f;      // flip so gyro sign matches d(theta)/dt

// Trim for IMU mounting tilt and COM position -- NOT for accel bias, which
// the offsets above now handle. Set it from the physical balance point:
// balance the bot by hand until it is equally willing to fall either way,
// read th, and put that value here in radians.
const float THETA_OFFSET = 0.0f;    // rad   <-- STILL TO BE MEASURED

const float ALPHA = 0.98f;          // complementary filter weight on gyro

// ============================================================
// CONTROL CONSTANTS  -- from Python/lqr.py
// ============================================================
// >>> EVERYTHING BELOW IS DISCRETE-TIME, DESIGNED AT EXACTLY THIS DT. <<<
// Change DT and these numbers are WRONG -- rerun lqr.py to regenerate.
//
// Why discrete and not forward Euler: the plant carries a back-EMF pole near
// -313 rad/s. Euler-integrating the observer at 5 ms puts the closed-loop
// spectral radius at 2.58 -- violently unstable -- even though the continuous
// design is fine. ZOH-discretizing first removes the integration error
// entirely and lands at |z| = 0.996.
const float DT = 0.005f;            // 200 Hz

// Discrete LQR gains, rho = 10.  Closed loop |z| = 0.985.
const float K_THETA = 14.0999f;     // V per rad
const float K_RATE  =  1.2094f;     // V per rad/s
const float K_WHEEL = -0.4529f;     // V per rad/s of wheel (velocity damping)

// ZOH-discretized plant Ad (3x3), Bd (3x1) for [theta, thetaDot, phiDot]
const float Ad00 = 1.002045f, Ad01 =  0.004238f, Ad02 = -0.000766f;
const float Ad10 = 0.708052f, Ad11 =  0.757039f, Ad12 = -0.245007f;
const float Ad20 = 0.715590f, Ad21 = -0.544071f, Ad22 =  0.453525f;
const float Bd0  = 0.002735f, Bd1  =  0.875024f, Bd2  =  1.951695f;

// Discrete Kalman predictor gain L (3x2), w_phid = 0.1
const float L00 = 0.001365f, L01 =  0.011871f;
const float L10 = 0.000900f, L11 =  0.555636f;
const float L20 = 0.001585f, L21 = -0.576621f;

// ============================================================
// DRIVE
// ============================================================
const float V_MOTOR_MAX  = 5.4f;    // pack volts MINUS driver drop (L298N ~2 V)

// MEASURED via MOTOR_TEST: both wheels break away at 110, same direction.
// That is 43% of full scale -- very high, and it is the reason the motors
// buzzed without turning while the deadband was still set to 45.
//
// A breakaway this large is mostly the L298N. It is a bipolar-transistor
// bridge that drops 1.5-2.5 V across its outputs, so at low duty cycle most
// of the applied voltage never reaches the motor. A MOSFET driver
// (TB6612FNG, DRV8833) drops ~0.5 V and would typically cut this to 50-60.
//
// Note: if MOTOR_TEST was run with the wheels off the ground, the on-ground
// value under the bot's own weight will be HIGHER. Re-check loaded.
const int   PWM_DEADBAND = 110;
const float SAFE_ANGLE   = 0.52f;   // ~30 deg -- linearization is void past this

// Re-arm conditions after a safety latch: held near vertical and steady for
// about a second. Deliberate enough that the bot never lurches in your hands.
const float REARM_ANGLE = 0.17f;    // ~10 deg
const float REARM_RATE  = 0.5f;     // rad/s
const int   REARM_TICKS = 200;      // 200 * 5 ms = 1 s

// ============================================================
// STATE
// ============================================================
float xhat[3] = {0, 0, 0};          // theta, thetaDot, phiDot (estimated)
float thetaComp = 0;                // complementary-filter angle
float gyroBias = 0;                 // rad/s
bool  latchedOff = false;
int   rearmCount = 0;
unsigned long lastMicros = 0;

// ============================================================
// IMU HELPERS
// ============================================================
void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission(true);
}

// Reads all 14 bytes into a buffer FIRST. Do not inline the reads as
// (Wire.read() << 8) | Wire.read() -- the evaluation order of the two calls
// is unspecified in C++, so the compiler is free to swap your bytes.
bool readIMURaw(float &accX, float &accY, float &accZ, float &gyroY) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_ACCEL_XOUT);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, 14, true) != 14) return false;

  uint8_t b[14];
  for (int i = 0; i < 14; i++) b[i] = Wire.read();

  int16_t rawAX = (int16_t)((b[0]  << 8) | b[1]);
  int16_t rawAY = (int16_t)((b[2]  << 8) | b[3]);
  int16_t rawAZ = (int16_t)((b[4]  << 8) | b[5]);
  int16_t rawGY = (int16_t)((b[10] << 8) | b[11]);   // b[8..9] = gyro X

  accX = rawAX / ACCEL_LSB;
  accY = rawAY / ACCEL_LSB;
  accZ = rawAZ / ACCEL_LSB;
  gyroY = (rawGY / GYRO_LSB) * (PI / 180.0f);        // rad/s
  return true;
}

// Same read, with the offset/scale correction applied.
bool readIMU(float &accX, float &accY, float &accZ, float &gyroY) {
  if (!readIMURaw(accX, accY, accZ, gyroY)) return false;
  accX = (accX - ACC_OFF_X) / ACC_SCL_X;
  accY = (accY - ACC_OFF_Y) / ACC_SCL_Y;
  accZ = (accZ - ACC_OFF_Z) / ACC_SCL_Z;
  return true;
}

// Six-position calibration. Each axis is pointed up, then down; the average
// of the two readings is that axis's offset, half their difference is its
// scale. Runs on RAW values, so the constants above must still be 0/1 when
// you run this.
void calibrateAccel() {
  const char *names[6] = {"+X up", "-X up", "+Y up", "-Y up", "+Z up", "-Z up"};
  float rx[6], ry[6], rz[6];

  Serial.println("\n=== ACCEL CALIBRATION ===");
  Serial.println("Rest the bot on a flat table so the named axis points UP.");
  Serial.println("Rough alignment is fine. Hold still, then send any char.\n");

  for (int pnt = 0; pnt < 6; pnt++) {
    Serial.print("Position "); Serial.print(pnt + 1);
    Serial.print("/6:  "); Serial.print(names[pnt]);
    Serial.println("   -> send a char when steady");

    while (Serial.available()) Serial.read();
    while (!Serial.available()) delay(10);
    while (Serial.available()) Serial.read();

    float sx = 0, sy = 0, sz = 0, gy;
    int n = 0;
    for (int i = 0; i < 200; i++) {
      float ax, ay, az;
      if (readIMURaw(ax, ay, az, gy)) { sx += ax; sy += ay; sz += az; n++; }
      delay(5);
    }
    rx[pnt] = sx / n; ry[pnt] = sy / n; rz[pnt] = sz / n;
    Serial.print("   ax="); Serial.print(rx[pnt], 4);
    Serial.print(" ay="); Serial.print(ry[pnt], 4);
    Serial.print(" az="); Serial.println(rz[pnt], 4);
  }

  float offX = (rx[0] + rx[1]) / 2, sclX = (rx[0] - rx[1]) / 2;
  float offY = (ry[2] + ry[3]) / 2, sclY = (ry[2] - ry[3]) / 2;
  float offZ = (rz[4] + rz[5]) / 2, sclZ = (rz[4] - rz[5]) / 2;

  Serial.println("\n--- paste into balance.ino, then set CALIBRATE 0 ---");
  Serial.print("const float ACC_OFF_X = "); Serial.print(offX, 5);
  Serial.print("f,  ACC_OFF_Y = "); Serial.print(offY, 5);
  Serial.print("f,  ACC_OFF_Z = "); Serial.print(offZ, 5); Serial.println("f;");
  Serial.print("const float ACC_SCL_X = "); Serial.print(sclX, 5);
  Serial.print("f,  ACC_SCL_Y = "); Serial.print(sclY, 5);
  Serial.print("f,  ACC_SCL_Z = "); Serial.print(sclZ, 5); Serial.println("f;");
  Serial.println("\n(scales should land near 1.0; a value far off suggests");
  Serial.println(" the breakout is not a genuine MPU-6050)");
}

// ============================================================
// MOTORS
// ============================================================
void setMotor(int pwmPin, int dir1, int dir2, int cmd) {
  if (cmd >= 0) {
    digitalWrite(dir1, HIGH);
    digitalWrite(dir2, LOW);
  } else {
    digitalWrite(dir1, LOW);
    digitalWrite(dir2, HIGH);
    cmd = -cmd;
  }
  analogWrite(pwmPin, cmd);
}

void stopMotors() {
  analogWrite(MOTOR_L_PWM, 0);
  analogWrite(MOTOR_R_PWM, 0);
}

// Map [1,255] onto [DEADBAND,255] so small commands actually turn the wheels.
int applyDeadband(int cmd) {
  if (cmd == 0) return 0;
  int s = (cmd > 0) ? 1 : -1;
  int mag = abs(cmd);
  return s * (PWM_DEADBAND + (mag * (255 - PWM_DEADBAND)) / 255);
}

int voltsToPwm(float V) {
  int pwm = (int)(255.0f * V / V_MOTOR_MAX);
  return constrain(pwm, -255, 255);
}

// Ramps each motor alone, then both together. Answers three things that
// have to be known before the loop is closed:
//   1. the PWM at which each wheel actually starts turning under load
//      -> PWM_DEADBAND
//   2. whether both wheels drive the bot the SAME way. They are mounted
//      mirror-image, so identical wiring turns them in opposite directions;
//      the bot then sits still while both motors strain and buzz.
//   3. whether the driver and battery can source the current at all.
void motorTest() {
  Serial.println("\n=== MOTOR TEST ===");
  Serial.println("Wheels OFF THE GROUND. Watch each wheel.");
  Serial.println("Note the PWM where it FIRST turns, and which way.\n");

  const char *phase[3] = {"LEFT only", "RIGHT only", "BOTH together"};
  for (int ph = 0; ph < 3; ph++) {
    Serial.print("\n--- "); Serial.print(phase[ph]); Serial.println(" ---");
    for (int pwm = 0; pwm <= 255; pwm += 10) {
      if (ph == 0 || ph == 2)
        setMotor(MOTOR_L_PWM, MOTOR_L_DIR1, MOTOR_L_DIR2, pwm);
      if (ph == 1 || ph == 2)
        setMotor(MOTOR_R_PWM, MOTOR_R_DIR1, MOTOR_R_DIR2, pwm);
      Serial.print("  pwm = "); Serial.println(pwm);
      delay(700);
    }
    stopMotors();
    delay(1000);
  }

  Serial.println("\n--- reverse, BOTH ---");
  for (int pwm = 0; pwm >= -255; pwm -= 10) {
    setMotor(MOTOR_L_PWM, MOTOR_L_DIR1, MOTOR_L_DIR2, pwm);
    setMotor(MOTOR_R_PWM, MOTOR_R_DIR1, MOTOR_R_DIR2, pwm);
    Serial.print("  pwm = "); Serial.println(pwm);
    delay(700);
  }
  stopMotors();
  Serial.println("\nDone. Set PWM_DEADBAND to the higher of the two");
  Serial.println("break-away values, then MOTOR_TEST 0.");
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(2000);                       // does not block when battery powered

  Wire.begin();
  Wire.setClock(400000);             // 14-byte read must fit inside 5 ms
  writeReg(REG_PWR_MGMT_1, 0x00);    // wake
  writeReg(REG_GYRO_CONFIG, GYRO_FS_SEL);
  delay(100);

  pinMode(MOTOR_L_PWM, OUTPUT);  pinMode(MOTOR_L_DIR1, OUTPUT);
  pinMode(MOTOR_L_DIR2, OUTPUT); pinMode(MOTOR_R_PWM, OUTPUT);
  pinMode(MOTOR_R_DIR1, OUTPUT); pinMode(MOTOR_R_DIR2, OUTPUT);
  stopMotors();

  // --- is the IMU even talking? ---
  // A silent I2C failure makes every reading zero, which then looks like a
  // control problem instead of a wiring problem. Fail loudly instead.
  Wire.beginTransmission(MPU_ADDR);
  if (Wire.endTransmission() != 0) {
    Serial.println("*** NO RESPONSE FROM MPU-6050 AT 0x68 ***");
    Serial.println("Check SDA/SCL/VCC/GND. AD0 low = 0x68, high = 0x69.");
    while (1) { stopMotors(); delay(500); }
  }

#if CALIBRATE
  calibrateAccel();
  while (1) { stopMotors(); delay(1000); }
#endif

#if MOTOR_TEST
  motorTest();
  while (1) { stopMotors(); delay(1000); }
#endif

  // --- gyro bias: hold still ---
  Serial.println("Calibrating gyro -- hold still");
  float accX = 0, accY = 0, accZ = 0, gyroY = 0, sum = 0;
  int n = 0, fails = 0;
  for (int i = 0; i < 1000; i++) {
    if (readIMU(accX, accY, accZ, gyroY)) { sum += gyroY; n++; }
    else fails++;
    delay(2);
  }
  if (n == 0) {
    Serial.println("*** IMU ADDRESSABLE BUT EVERY READ FAILED ***");
    while (1) { stopMotors(); delay(500); }
  }
  if (fails > 0) {
    Serial.print("WARNING: "); Serial.print(fails);
    Serial.println("/1000 reads failed -- flaky wiring");
  }
  gyroBias = sum / n;
  Serial.print("gyro bias (rad/s): "); Serial.println(gyroBias, 5);

  // --- seed theta from gravity so the filter does not start at a lie ---
  if (readIMU(accX, accY, accZ, gyroY)) {
    thetaComp = THETA_SIGN * atan2(ACC_FOREAFT, ACC_UP) - THETA_OFFSET;
  }
  xhat[0] = thetaComp;
  xhat[1] = 0;
  xhat[2] = 0;

  // One-shot raw dump so the axis mapping can be confirmed without
  // streaming. Standing upright, the axis used for ACC_UP must read ~+1.0
  // and the other two ~0.0.
  Serial.print("resting accel  aX="); Serial.print(accX, 3);
  Serial.print("  aY="); Serial.print(accY, 3);
  Serial.print("  aZ="); Serial.println(accZ, 3);
  Serial.print("static |a| = ");
  Serial.println(sqrt(accX*accX + accY*accY + accZ*accZ), 3);
  Serial.println("(should be ~1.000 in EVERY orientation, not just one)");

  Serial.print("resting theta = ");
  Serial.print(thetaComp * 180.0f / PI, 1);
  Serial.println(" deg");

#if !OPEN_LOOP
  // Guard against closing the loop before THETA_OFFSET is measured.
  // If the bot reads a large angle while sitting still, the controller
  // commands large voltage immediately, drives itself past SAFE_ANGLE,
  // and latches -- which looks exactly like "the motors are broken".
  if (fabs(thetaComp) > REARM_ANGLE) {
    Serial.print("\n*** REFUSING TO CLOSE THE LOOP -- theta = ");
    Serial.print(thetaComp * 180.0f / PI, 1);
    Serial.println(" deg at rest ***");
    Serial.print("Closed loop would command ");
    Serial.print(fabs(K_THETA * thetaComp), 2);
    Serial.println(" V while sitting still.");
    if (fabs(thetaComp) > 1.0f) {          // ~57 deg: not a trim problem
      Serial.println("That is nearly horizontal -- the bot is lying down.");
      Serial.println("Stand it upright and hold it STILL through startup;");
      Serial.println("the gyro calibrates for ~2 s before theta is seeded.");
    } else {
      Serial.println("THETA_OFFSET has not been set from the balance point.");
      Serial.println("Set OPEN_LOOP 1, do bring-up steps 1-3, then come back.");
    }
    while (1) { stopMotors(); delay(1000); }
  }
#endif

  lastMicros = micros();
}

// ============================================================
// LOOP  -- fixed rate
// ============================================================
void loop() {
  unsigned long now = micros();
  if (now - lastMicros < (unsigned long)(DT * 1e6f)) return;
  lastMicros += (unsigned long)(DT * 1e6f);

  float accX, accY, accZ, gyroY;
  if (!readIMU(accX, accY, accZ, gyroY)) { stopMotors(); return; }

  // ---- measurements ----
  float thetaDot = GYRO_SIGN * (gyroY - gyroBias);
  float thetaAcc = THETA_SIGN * atan2(ACC_FOREAFT, ACC_UP) - THETA_OFFSET;

  // Complementary filter. High ALPHA because the accel measures total
  // specific force -- it tilts under the bot's own acceleration and lies
  // exactly when a correction is underway.
  thetaComp = ALPHA * (thetaComp + thetaDot * DT) + (1.0f - ALPHA) * thetaAcc;

  // ---- safety ----
  // Announce the latch. A silent cutoff is indistinguishable from dead
  // motors, a bad driver, or a flat battery -- say which it is.
  if (!latchedOff && fabs(thetaComp) > SAFE_ANGLE) {
    latchedOff = true;
    Serial.print("\n*** SAFETY LATCH: |theta| = ");
    Serial.print(fabs(thetaComp) * 180.0f / PI, 1);
    Serial.print(" deg > ");
    Serial.print(SAFE_ANGLE * 180.0f / PI, 0);
    Serial.println(" deg -- motors off ***");
    Serial.println("    Hold it upright and still for 1 s to re-arm.");
  }
  if (latchedOff) {
    stopMotors();
    xhat[0] = thetaComp; xhat[1] = thetaDot; xhat[2] = 0;   // keep sane
    if (fabs(thetaComp) < REARM_ANGLE && fabs(thetaDot) < REARM_RATE) {
      if (++rearmCount >= REARM_TICKS) {
        latchedOff = false;
        rearmCount = 0;
        Serial.println("*** re-armed ***");
      }
    } else {
      rearmCount = 0;
    }
    return;
  }

  // ---- control, from the CURRENT estimate ----
  float V = -(K_THETA * xhat[0] + K_RATE * xhat[1] + K_WHEEL * xhat[2]);
  V = constrain(V, -V_MOTOR_MAX, V_MOTOR_MAX);

  // The observer must be fed the voltage ACTUALLY APPLIED, never the
  // pre-saturation command. In OPEN_LOOP the motors are off, so that is 0 --
  // otherwise the estimator integrates a phantom 5.4 V and phiDot runs away.
#if OPEN_LOOP
  float V_applied = 0.0f;
#else
  float V_applied = V;
#endif

  // ---- observer: xhat[k+1] = Ad xhat + Bd V + L(y - C xhat) ----
  // No DT anywhere -- the timestep is already baked into Ad, Bd and L.
  float e0 = thetaComp - xhat[0];        // innovation on theta
  float e1 = thetaDot  - xhat[1];        // innovation on thetaDot

  float n0 = Ad00 * xhat[0] + Ad01 * xhat[1] + Ad02 * xhat[2] + Bd0 * V_applied
             + L00 * e0 + L01 * e1;
  float n1 = Ad10 * xhat[0] + Ad11 * xhat[1] + Ad12 * xhat[2] + Bd1 * V_applied
             + L10 * e0 + L11 * e1;
  float n2 = Ad20 * xhat[0] + Ad21 * xhat[1] + Ad22 * xhat[2] + Bd2 * V_applied
             + L20 * e0 + L21 * e1;

  xhat[0] = n0;
  xhat[1] = n1;
  xhat[2] = n2;

  // ---- drive ----
  int cmd = applyDeadband(voltsToPwm(V));

#if OPEN_LOOP
  stopMotors();
  static int dec = 0;
  if (++dec >= 20) {                     // 10 Hz print
    dec = 0;
    Serial.print("th="); Serial.print(thetaComp * 180.0f / PI, 1);
    Serial.print(" thd="); Serial.print(thetaDot, 2);
    Serial.print(" phid_est="); Serial.print(xhat[2], 2);
    Serial.print(" V="); Serial.print(V, 2);
    Serial.print(" pwm="); Serial.println(cmd);
  }
#else
  setMotor(MOTOR_L_PWM, MOTOR_L_DIR1, MOTOR_L_DIR2, cmd);
  setMotor(MOTOR_R_PWM, MOTOR_R_DIR1, MOTOR_R_DIR2, cmd);
#endif
}
