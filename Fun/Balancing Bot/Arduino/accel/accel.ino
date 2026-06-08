#include <Wire.h>
const int MPU_ADDR = 0x68;

int16_t rawAccX, rawAccY, rawAccZ;
int16_t rawGyroX, rawGyroY, rawGyroZ;
int16_t rawTemp;

float gyroXOffset = 0;
float gyroYOffset = 0;
float gyroZOffset = 0;

void setup() {
  
  Serial.begin(115200);
  while (!Serial);             // Wait for Serial Monitor to open (critical on Arduino Uno R4)
  Wire.begin();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0); // Wakes up the device
  Wire.endTransmission(true);
  Serial.println("Calibrating Gryo.. Do not move the sensor");

  long sumX = 0, sumY = 0, sumZ = 0;
  int samples = 2000;

  for (int i=0; i<samples; i++) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x43);
    Wire.endTransmission(false);
    Wire.requestFrom(MPU_ADDR, 6, true);

    sumX += (int16_t)(Wire.read() << 8) | Wire.read();
    sumY += (int16_t)(Wire.read() << 8) | Wire.read();
    sumZ += (int16_t)(Wire.read() << 8) | Wire.read();

    gyroXOffset = (float)sumX/ samples;
    gyroYOffset = (float)sumY/ samples;
    gyroZOffset = (float)sumZ/ samples;


    

  }
  Serial.println("Calibration complete");
}
void loop() {
  //Point to the starting register
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false); //keeps I2C bus active

  //Request 14 bytes of data (the 7 values above)
  Wire.requestFrom(MPU_ADDR, 14, true); //true releases the bus when done

  // Read and combine high and low bytes for accelerometer

  rawAccX = (Wire.read() << 8) | Wire.read();
  rawAccY = (Wire.read() << 8) | Wire.read();
  rawAccZ = (Wire.read() << 8) | Wire.read();

  rawTemp = (Wire.read() << 8) | Wire.read();

  rawGyroX = (Wire.read() << 8) | Wire.read();
  rawGyroY = (Wire.read() << 8) | Wire.read();
  rawGyroZ = (Wire.read() << 8) | Wire.read();

  float gyroX = ((float)rawGyroX - gyroXOffset) / 131.0;
  float gyroY = ((float)rawGyroY - gyroYOffset) / 131.0;
  float gyroZ = ((float)rawGyroZ - gyroZOffset) / 131.0;

  float accX = (float)rawAccX / 16384.0;
  float accY = (float)rawAccY / 16384.0;
  float accZ = (float)rawAccZ / 16384.0;

  float temp = ((float)rawTemp / 340.0) + 36.53;


  Serial.print("AccX: "); Serial.println(accX);
  Serial.print("AccY: "); Serial.println(accY);
  Serial.print("AccZ: "); Serial.println(accZ);

  Serial.print("GyroX: "); Serial.println(gyroX);
  Serial.print("GyroY: "); Serial.println(gyroY);
  Serial.print("GyroZ: "); Serial.println(gyroZ);

  Serial.print("Temp: "); Serial.println(temp);

  delay(100);

 
}