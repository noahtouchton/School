/*
 * Simple Echo Sketch for Python Integration
 * * 1. Receives a string ending in newline (\n)
 * 2. Prints it back (for debugging)
 * 3. Sends "Finished" to release the Python lock
 */

const int BAUD_RATE = 9600; // Must match your Python code
const int LED_PIN = LED_BUILTIN; // Visual feedback


void process_command(String cmd, float *result) {
  int commaIndex = cmd.indexOf(',');
  if (commaIndex != -1) { 
    
    String x_str = cmd.substring(0, commaIndex);
    
    String y_str = cmd.substring(commaIndex + 1);

    result[0] = x_str.toFloat();
    result[1] = y_str.toFloat();
  }
}



void setup() {
  Serial.begin(BAUD_RATE);
  pinMode(LED_PIN, OUTPUT);
  
  // Optional: Flash LED 3 times on startup to show it reset
  for(int i=0; i<3; i++) {
    digitalWrite(LED_PIN, HIGH); delay(100);
    digitalWrite(LED_PIN, LOW); delay(100);
  }
}

void loop() {
  // Check if data is available in the serial buffer
  if (Serial.available() > 0) {
    
    // Read the incoming command until the newline character
    // Python sends: "YOUR_COMMAND\n"
    String received_cmd = Serial.readStringUntil('\n');
    
    // Remove any extra whitespace or \r characters
    received_cmd.trim();

    // --- DO YOUR WORK HERE ---
    float *data;

    process_command(received_cmd, data);

    
    // 1. Flash LED to show we got something
    digitalWrite(LED_PIN, HIGH);

    // 2. Echo the command back. 
    // The Python Worker will capture this and emit it via the .log signal.

    
    Serial.print("x: ");
    Serial.print(data[0]);
    Serial.print(" y: ");
    Serial.println(data[1]);

    // simulate a tiny bit of work (optional)
    delay(1000); 
    
    digitalWrite(LED_PIN, LOW);

    // --- END OF WORK ---

    // 3. CRITICAL: Send "Finished".
    // If you delete this line, Python will hang for 5 seconds (timeout).
    Serial.println("Finished");
  }
}