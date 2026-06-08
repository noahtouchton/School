#include <Wire.h>
#include <WiFiS3.h>
// ==========================================
// WIFI CREDENTIALS (CHANGE THESE!)
// ==========================================
char ssid[] = "TreeFamilyWifi";
char pass[] = "HOME54321";
// ==========================================
// PHYSICAL CONSTANTS (from user parameters)
// ==========================================
const float kt = 0.28;      // Torque constant (N*m/A)
const float R = 5.5;        // Motor resistance (Ohms)
const int MPU_ADDR = 0x68;  // I2C address of MPU-6050
// ==========================================
// MOTOR DRIVER PINS (CHANGE TO MATCH YOUR WIRING)
// ==========================================
const int MOTOR_L_PWM = 5;
const int MOTOR_L_DIR1 = 3;
const int MOTOR_L_DIR2 = 4;
const int MOTOR_R_PWM = 6;
const int MOTOR_R_DIR1 = 7;
const int MOTOR_R_DIR2 = 8;
// Web Server
WiFiServer server(80);
struct Sample {
  unsigned long timeUs;
  float rateRad;
};
const int MAX_SAMPLES = 30;
Sample samples[MAX_SAMPLES];
void setup() {
  Serial.begin(115200);
  delay(2000); // 2-second delay so you have time to open Serial Monitor (doesn't block if powered by battery)

  // Initialize I2C and wake up MPU-6050
  Wire.begin();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);

  // Set motor pins as outputs
  pinMode(MOTOR_L_PWM, OUTPUT);
  pinMode(MOTOR_L_DIR1, OUTPUT);
  pinMode(MOTOR_L_DIR2, OUTPUT);
  pinMode(MOTOR_R_PWM, OUTPUT);
  pinMode(MOTOR_R_DIR1, OUTPUT);
  pinMode(MOTOR_R_DIR2, OUTPUT);
  stopMotors();

    // Connect to WiFi
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  
  // Wait for DHCP to assign an actual IP address
  while (WiFi.localIP() == IPAddress(0, 0, 0, 0)) {
    delay(500);
    Serial.print("?");
  }
  Serial.println();

  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  server.begin();
}
void loop() {
  WiFiClient client = server.available();
  if (client) {
    Serial.println("New client connected.");
    String currentLine = "";
    float batVoltage = 7.4;
    int motorPWM = 150;
    boolean runTest = false;
    while (client.connected()) {
      if (client.available()) {
        char c = client.read();
        if (c == '\n') {
          if (currentLine.length() == 0) {
            // Send standard HTTP headers
            client.println("HTTP/1.1 200 OK");
            client.println("Content-Type: text/html");
            client.println("Connection: close");
            client.println();
            if (runTest) {
              // Run the trial
              float calculatedIyy = runInertiaTrial(batVoltage, motorPWM);
              sendResultPage(client, calculatedIyy, batVoltage, motorPWM);
            } else {
              sendConfigPage(client);
            }
            break;
          } else {
            if (currentLine.indexOf("GET /start") != -1) {
              runTest = true;
              // Parse voltage
              int voltIdx = currentLine.indexOf("voltage=");
              if (voltIdx != -1) {
                int nextAmp = currentLine.indexOf("&", voltIdx);
                String voltStr = currentLine.substring(voltIdx + 8, nextAmp != -1 ? nextAmp : currentLine.length());
                batVoltage = voltStr.toFloat();
              }
              // Parse PWM
              int pwmIdx = currentLine.indexOf("pwm=");
              if (pwmIdx != -1) {
                int nextSpace = currentLine.indexOf(" ", pwmIdx);
                String pwmStr = currentLine.substring(pwmIdx + 4, nextSpace != -1 ? nextSpace : currentLine.length());
                motorPWM = pwmStr.toInt();
              }
            }
            currentLine = "";
          }
        } else if (c != '\r') {
          currentLine += c;
        }
      }
    }
    client.stop();
    Serial.println("Client disconnected.");
  }
}
// Controls motor speed and direction
void setMotorsPWM(int pwm) {
  // Drive forward
  digitalWrite(MOTOR_L_DIR1, HIGH);
  digitalWrite(MOTOR_L_DIR2, LOW);
  digitalWrite(MOTOR_R_DIR1, HIGH);
  digitalWrite(MOTOR_R_DIR2, LOW);
  analogWrite(MOTOR_L_PWM, pwm);
  analogWrite(MOTOR_R_PWM, pwm);
}
void stopMotors() {
  analogWrite(MOTOR_L_PWM, 0);
  analogWrite(MOTOR_R_PWM, 0);
}
// Read raw MPU-6050 Gyro Y axis (pitch axis)
float readGyroYRadS() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x45); // Gyro Y register high byte
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 2, true);
  
  int16_t rawGyroY = (int16_t)((Wire.read() << 8) | Wire.read());
  float gyroY = (float)rawGyroY / 131.0; // Convert to deg/s
  return gyroY * PI / 180.0;            // Convert to rad/s
}
// Runs the dynamic acceleration pulse and returns Iyy
float runInertiaTrial(float batVoltage, int motorPWM) {
  Serial.println("Starting Trial in 2 seconds... HOLD WHEELS STILL!");
  delay(2000);
  // Calibrate Gyro offset (standing still)
  float offsetSum = 0;
  for (int i = 0; i < 50; i++) {
    offsetSum += readGyroYRadS();
    delay(5);
  }
  float gyroOffset = offsetSum / 50.0;
  Serial.println("GO!");
  unsigned long startUs = micros();
  setMotorsPWM(motorPWM);
  for (int i = 0; i < MAX_SAMPLES; i++) {
    samples[i].timeUs = micros() - startUs;
    samples[i].rateRad = readGyroYRadS() - gyroOffset;
    delayMicroseconds(4000); // ~5ms loop time with I2C overhead
  }
  stopMotors();
  Serial.println("Trial Finished!");
  // Linear Regression (Least-Squares) on first 5 samples (~25ms) to find alpha_0
  int regressionSamples = 5;
  float sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  for (int i = 0; i < regressionSamples; i++) {
    float t = (float)samples[i].timeUs / 1000000.0; // seconds
    sumX += t;
    sumY += samples[i].rateRad;
    sumXY += t * samples[i].rateRad;
    sumXX += t * t;
  }
  float alpha_0 = (regressionSamples * sumXY - sumX * sumY) / (regressionSamples * sumXX - sumX * sumX);
  alpha_0 = abs(alpha_0); // Take absolute acceleration
  // Calculate Applied Voltage
  float appliedVoltage = batVoltage * (motorPWM / 255.0);
  // Calculate Moment of Inertia Iyy = (kt * V) / (R * alpha_0)
  float Iyy = (kt * appliedVoltage) / (R * alpha_0);
  Serial.print("Initial Accel (alpha_0): "); Serial.print(alpha_0); Serial.println(" rad/s^2");
  Serial.print("Applied Voltage: "); Serial.print(appliedVoltage); Serial.println(" V");
  Serial.print("Calculated Iyy: "); Serial.print(Iyy, 6); Serial.println(" kg*m^2");
  return Iyy;
}
// Config HTML Page
void sendConfigPage(WiFiClient client) {
  client.println("<!DOCTYPE html><html><head><meta name='viewport' content='width=device-width, initial-scale=1'>");
  client.println("<style>body{font-family:sans-serif; background:#121212; color:#fff; text-align:center; padding:20px;}");
  client.println(".card{background:#1e1e1e; border-radius:15px; padding:30px; display:inline-block; box-shadow:0 10px 20px rgba(0,0,0,0.5);}");
  client.println("input{background:#2a2a2a; border:none; border-radius:5px; color:#fff; padding:10px; margin:10px; width:80px; text-align:center;}");
  client.println("button{background:#00bcd4; border:none; border-radius:5px; color:#fff; padding:12px 24px; font-weight:bold; cursor:pointer;}");
  client.println("h2{color:#00bcd4;}</style></head><body>");
  client.println("<div class='card'><h2>Iyy Inertia Calculator</h2>");
  client.println("<p><b>IMPORTANT:</b> Place the chassis suspended from its axles (wheels removed). Hold the axles firmly still so only the chassis rotates.</p>");
  client.println("<form action='/start' method='get'>");
  client.println("Battery Voltage (V): <input type='number' name='voltage' step='0.1' value='7.4'><br>");
  client.println("Test PWM (50-200): <input type='number' name='pwm' value='120'><br><br>");
  client.println("<button type='submit'>Start Trial</button>");
  client.println("</form></div></body></html>");
}
// Result HTML Page
void sendResultPage(WiFiClient client, float Iyy, float voltage, int pwm) {
  client.println("<!DOCTYPE html><html><head><meta name='viewport' content='width=device-width, initial-scale=1'>");
  client.println("<style>body{font-family:sans-serif; background:#121212; color:#fff; text-align:center; padding:20px;}");
  client.println(".card{background:#1e1e1e; border-radius:15px; padding:30px; display:inline-block;}");
  client.println("h2{color:#4caf50;} .val{font-size:32px; font-weight:bold; color:#ffeb3b; margin:15px 0;}");
  client.println(".btn{background:#00bcd4; text-decoration:none; color:#fff; padding:10px 20px; border-radius:5px; display:inline-block; margin-top:20px;}</style></head><body>");
  client.println("<div class='card'><h2>Trial Completed Successfully</h2>");
  client.println("<p>Calculated Moment of Inertia (Iyy):</p>");
  client.println("<div class='val'>" + String(Iyy, 6) + " kg&middot;m<sup>2</sup></div>");
  client.println("<p><b>Details:</b><br>Battery Voltage: " + String(voltage, 1) + " V<br>Test PWM: " + String(pwm) + "</p>");
  
  // Show table of data points
  client.println("<h3>Logged Data:</h3><table style='margin:0 auto; text-align:left; border-collapse:collapse;'>");
  client.println("<tr><th style='padding:5px 15px;'>Time (ms)</th><th style='padding:5px 15px;'>Gyro Y (rad/s)</th></tr>");
  for (int i = 0; i < MAX_SAMPLES; i++) {
    client.println("<tr><td style='padding:2px 15px;'>" + String(samples[i].timeUs / 1000.0, 1) + "</td>");
    client.println("<td style='padding:2px 15px;'>" + String(samples[i].rateRad, 4) + "</td></tr>");
  }
  client.println("</table>");
  
  client.println("<a href='/' class='btn'>Configure New Trial</a></div></body></html>");
}
