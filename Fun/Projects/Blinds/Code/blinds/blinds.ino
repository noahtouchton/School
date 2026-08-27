#include <Wire.h>
#include <math.h>

#define AS5600_ADDR 0x36

// Register addresses (Chunk 2)
#define REG_STATUS     0x0B
#define REG_ANGLE_HIGH 0x0E  // filtered angle, high byte
#define REG_ANGLE_LOW  0x0F  // filtered angle, low byte

#define IN1 D1
#define IN2 D2
#define ENA D3

#define Kp 1.0


void setup() {
  setupWifi();
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);
  Wire.begin(D4, D5); // SDA, SCL
  Serial.begin(115200);
  delay(1000);
  Serial.println("I2C Scanner starting...");
}

bool scanI2C() {
  byte count = 0;
  for (byte addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("Found device at 0x");
      Serial.println(addr, HEX);
      count++;
    }
  }
  if (count == 0) Serial.println("No I2C devices found");
  return count > 0;
}

bool checkMagnet() {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(REG_STATUS);
  Wire.endTransmission(false);

  Wire.requestFrom(AS5600_ADDR, 1);
  uint8_t status = Wire.read();

  bool magnetDetected = !(status & 0x20); // MD bit is bit 5

  return magnetDetected;
}

uint16_t readAngleRaw() {
  Wire.beginTransmission(AS5600_ADDR);
  Wire.write(REG_ANGLE_HIGH);
  Wire.endTransmission(false);

  Wire.requestFrom(AS5600_ADDR, 2);
  uint16_t highByte = Wire.read();
  uint16_t lowByte = Wire.read();

  uint16_t angle = (highByte << 8) | lowByte;
  return angle;
}

float readAngleDegrees() {
  uint16_t rawAngle = readAngleRaw();
  return (rawAngle * 360.0) / 4096.0; // Convert to degrees
}

void setMotorSpeed(float speed) {
  // add minmum speed if required. Find lowest speed that makes the motor turn
  if (abs(speed)>255) {
    speed = 255 * (speed)/abs(speed);
  }
  if (speed > 0) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, speed);
  } else {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    analogWrite(ENA, abs(speed));
  }
}

int updateAngle(float oldRawAngle, float newRawAngle, int N) {
  float jump = newRawAngle - oldRawAngle;
  if (jump < -180){
    // on different N
    return N+1;
  } else if (jump > 180) {
    return N-1;
  } else {
    // on same N
    return N;
  }
}

void setPosition(float targetAngle, int N) {
  //target Angle will be given as full number not standard between 0 and 360  
  float startingAngle = readAngleDegrees();
  float currentRawAngle = startingAngle;
  float oldRawAngle = startingAngle;
  float angleDifference;
  float totalDistance;
  float currentAngle;
  int numRots = N;

  publishState(true);

  do {
    currentRawAngle = readAngleDegrees();
    numRots = updateAngle(oldRawAngle, currentRawAngle, numRots);
    currentAngle = currentRawAngle + 360.0*numRots;
    angleDifference = targetAngle - currentAngle;

    float speed = angleDifference * Kp;
    setMotorSpeed(speed);
    oldRawAngle = currentRawAngle;
    publishN(numRots);
  } while (abs(angleDifference) > 10);
  setMotorSpeed(0);
  
  publishState(false);
}

void loop() {
  loopWifi();
}