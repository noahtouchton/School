#include <AccelStepper.h>
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
const uint8_t X_MIN_LIMIT_PIN = 22;  // Long side of the bed (X axis)
const uint8_t Y_MIN_LIMIT_PIN = 23;  // Short side of the bed (Y axis)

// Maximum limit switches sit at the far end of each axis.
const uint8_t X_MAX_LIMIT_PIN = 24;  // Long side max (X axis)
const uint8_t Y_MAX_LIMIT_PIN = 25;  // Short side max (Y axis)

// Door limit switches - operations pause if either door is opened
const uint8_t DOOR_1_LIMIT_PIN = 26;  // Door 1 switch
const uint8_t DOOR_2_LIMIT_PIN = 27;  // Door 2 switch

// Set to true for switches that pull the pin LOW when pressed (recommended).
const bool LIMIT_ACTIVE_LOW = true;

// ---------------------- Grid Layout Configuration ----------------------
// Grid dimensions: 8 columns x 5 rows
const int GRID_COLUMNS = 8;
const int GRID_ROWS = 5;
const int GRID_TOTAL_POSITIONS = GRID_COLUMNS * GRID_ROWS;  // 40 positions

// Physical grid dimensions in inches
const float GRID_WIDTH_INCHES = 29.5f;   // X direction (long side)
const float GRID_HEIGHT_INCHES = 17.0f; // Y direction (short side)

// Calculate spacing between grid positions
// Convert inches to mm: 1 inch = 25.4 mm
const float GRID_COL_SPACING_MM = (GRID_WIDTH_INCHES / (GRID_COLUMNS - 1)) * 25.4f;   // Spacing between columns
const float GRID_ROW_SPACING_MM = (GRID_HEIGHT_INCHES / (GRID_ROWS - 1)) * 25.4f;      // Spacing between rows

// ---------------------- Deposit Grid Layout Configuration ----------------------
// Deposit grid dimensions: 5 rows x 8 columns (for depositing samples)
const int DEPOSIT_GRID_ROWS = 5;
const int DEPOSIT_GRID_COLUMNS = 8;
const int DEPOSIT_GRID_TOTAL_POSITIONS = DEPOSIT_GRID_ROWS * DEPOSIT_GRID_COLUMNS;  // 40 positions

// Physical deposit grid dimensions in mm
const float DEPOSIT_GRID_WIDTH_MM = 310.0f;   // X direction (long side)
const float DEPOSIT_GRID_HEIGHT_MM = 515.0f;  // Y direction (short side)

// Calculate spacing between deposit grid positions
const float DEPOSIT_GRID_COL_SPACING_MM = DEPOSIT_GRID_WIDTH_MM / (DEPOSIT_GRID_COLUMNS - 1);   // Spacing between columns
const float DEPOSIT_GRID_ROW_SPACING_MM = DEPOSIT_GRID_HEIGHT_MM / (DEPOSIT_GRID_ROWS - 1);     // Spacing between rows

// ---------------------- Sampling Position Placeholders ----------------------
// Fill these placeholders with real step offsets once measurements are known.
const long STEPS_TO_FIRST_POSITION_X = 5;
const long STEPS_TO_FIRST_POSITION_Y = 14;
const long CAM_POS_OFFSET_X_STEPS   = 0;
const long CAM_POS_OFFSET_Y_STEPS   = -10;
const long PICK_POS_OFFSET_X_STEPS  = 0;
const long PICK_POS_OFFSET_Y_STEPS  = 10;
const long DROP_POS_OFFSET_X_STEPS  = 0;
const long DROP_POS_OFFSET_Y_STEPS  = 0;
const long SMALL_OFFSET_STEPS       = 0;  // Tiny post-pickup move
const long SAMPLE_COL_SPACING_STEPS = 500;
const long SAMPLE_ROW_SPACING_STEPS = 200;
const int  SAMPLES_PER_ROW          = 5;
const unsigned long TOOLHEAD_SERIAL_BAUD = 115200UL;
const unsigned long HOST_SERIAL_TIMEOUT_MS = 60000UL;
const unsigned long TOOLHEAD_SERIAL_TIMEOUT_MS = 60000UL;

// ---------------------- Python Communication Configuration ----------------------
const unsigned long PYTHON_SERIAL_BAUD = 115200UL;
const unsigned long PYTHON_RESPONSE_TIMEOUT_MS = 30000UL;  // 30 second timeout for image capture
const String PYTHON_CAPTURE_COMMAND = "CAPTURE_IMAGE";
const String PYTHON_CAPTURE_RESPONSE = "IMAGE_CAPTURED";
const String PYTHON_READY_RESPONSE = "PYTHON_READY";

// ---------------------- Toolhead Communication Configuration ----------------------
const unsigned long TOOLHEAD_RESPONSE_TIMEOUT_MS = 60000UL;  // 60 second timeout for sample taking
const String TOOLHEAD_SAMPLE_COMMAND = "TAKE_SAMPLE";
const String TOOLHEAD_SAMPLE_RESPONSE = "SAMPLE_TAKEN";
const String TOOLHEAD_DEPOSIT_COMMAND = "DEPOSIT_SAMPLE";
const String TOOLHEAD_DEPOSIT_RESPONSE = "SAMPLE_DEPOSITED";
const String TOOLHEAD_READY_RESPONSE = "TOOLHEAD_READY";

// ---------------------- Sampling Motion Configuration ----------------------
// Y-axis offset to make room for toolhead after image capture
const float Y_OFFSET_FOR_TOOLHEAD_MM = 10.0f;  // Adjust this value based on your toolhead clearance needs

// ---------------------- Motion Tuning ----------------------
// Travel motion (general use after homing)
const float TRAVEL_MAX_SPEED = 1200.0f;  // steps per second
const float TRAVEL_ACCEL     = 800.0f;   // steps per second^2

// Homing motion
const float HOMING_SPEED     = 800.0f;   // approach speed toward the switch
const float HOMING_ACCEL     = 800.0f;   // acceleration during homing
const float BACKOFF_SPEED    = 300.0f;   // speed when backing away
const float BACKOFF_ACCEL    = 600.0f;   // acceleration when backing away
const long  BACKOFF_STEPS    = 80;     // steps to retreat after triggering

// ---------------------- Stepper Objects ----------------------
AccelStepper motorA(AccelStepper::DRIVER, MOTOR_A_STEP_PIN, MOTOR_A_DIR_PIN);
AccelStepper motorB(AccelStepper::DRIVER, MOTOR_B_STEP_PIN, MOTOR_B_DIR_PIN);

// ---------------------- Calibration Variables ----------------------
long maxXSteps = 0L;  // Will be set during calibration
long maxYSteps = 0L;  // Will be set during calibration
long totalSamples = 0L;           // User-provided total sample count
long firstSampleBaseXSteps = 0L;  // Camera XY for sample 0 (absolute steps)
long firstSampleBaseYSteps = 0L;
long firstDepositGridXSteps = 0L;  // X position (in steps) of deposit grid position 1
long firstDepositGridYSteps = 0L;  // Y position (in steps) of deposit grid position 1
bool samplingSequenceArmed = false;
bool samplingSequenceComplete = false;

// ---------------------- CoreXY Helper Functions ----------------------
// Motor and pulley specifications:
// - Nema 17 stepper: 1.8 degrees per step = 200 steps per revolution
// - GT2 belt: 2mm pitch
// - 20-tooth pulley: 20 teeth × 2mm = 40mm per revolution
// - Steps per mm: 200 steps / 40mm = 5.0 steps/mm
const float STEPS_PER_MM = 5.0f;

inline long coreXY_A(long xSteps, long ySteps) { return xSteps + ySteps; }
inline long coreXY_B(long xSteps, long ySteps) { return xSteps - ySteps; }
inline long currentX() {
  return (motorA.currentPosition() + motorB.currentPosition()) / 2;
}
inline long currentY() {
  return (motorA.currentPosition() - motorB.currentPosition()) / 2;
}
inline float absf(float value) { return (value >= 0.0f) ? value : -value; }

/**
 * Converts a physical distance in millimeters to the number of motor steps needed.
 * Works for both X and Y axes in CoreXY systems.
 * 
 * @param distanceMm The distance to travel in millimeters (can be positive or negative)
 * @return The number of steps needed (rounded to nearest integer)
 */
long distanceToSteps(float distanceMm) {
  return (long)round(distanceMm * STEPS_PER_MM);
}

/**
 * Navigates to a specific grid position in the 8x5 grid layout.
 * Grid positions are numbered sequentially from 1 to 40, starting at the top-left
 * and proceeding row by row (left to right, top to bottom).
 * 
 * Position numbering:
 *   1  2  3  4  5  6  7  8
 *   9 10 11 12 13 14 15 16
 *  17 18 19 20 21 22 23 24
 *  25 26 27 28 29 30 31 32
 *  33 34 35 36 37 38 39 40
 * 
 * @param gridPosition The grid position number (1-40, where 1 is top-left)
 * @param firstPositionXSteps The X position (in steps) of grid position 1
 * @param firstPositionYSteps The Y position (in steps) of grid position 1
 * @return true if successful, false if gridPosition is out of range
 */
bool navigateToGridPosition(int gridPosition, long firstPositionXSteps, long firstPositionYSteps) {
  // Validate grid position
  if (gridPosition < 1 || gridPosition > GRID_TOTAL_POSITIONS) {
    Serial.print(F("ERROR: Grid position "));
    Serial.print(gridPosition);
    Serial.print(F(" is out of range (1-"));
    Serial.print(GRID_TOTAL_POSITIONS);
    Serial.println(F(")"));
    return false;
  }
  
  // Convert 1-indexed position to 0-indexed
  int positionIndex = gridPosition - 1;
  
  // Calculate row and column (0-indexed)
  int column = positionIndex % GRID_COLUMNS;  // 0-7
  int row = positionIndex / GRID_COLUMNS;     // 0-4
  
  // Calculate physical offsets from first position
  float xOffsetMm = column * GRID_COL_SPACING_MM;
  float yOffsetMm = row * GRID_ROW_SPACING_MM;
  
  // Convert to steps
  long xSteps = firstPositionXSteps + distanceToSteps(xOffsetMm);
  long ySteps = firstPositionYSteps + distanceToSteps(yOffsetMm);
  
  // Debug output
  Serial.print(F("Navigating to grid position "));
  Serial.print(gridPosition);
  Serial.print(F(" (Row "));
  Serial.print(row + 1);
  Serial.print(F(", Column "));
  Serial.print(column + 1);
  Serial.print(F(") -> X: "));
  Serial.print(xSteps);
  Serial.print(F(" steps, Y: "));
  Serial.print(ySteps);
  Serial.println(F(" steps"));
  
  // Move to the calculated position
  // This calls moveToXYPosition() -> moveCoreXYRelative() which runs the motors
  // until they reach the target position (motorA.run() and motorB.run() are called)
  moveToXYPosition(xSteps, ySteps);
  
  return true;
}

/**
 * Navigates to a specific deposit grid position in the 5x8 deposit grid layout.
 * Deposit grid positions are numbered sequentially from 1 to 40, starting at the top-left
 * and proceeding row by row (left to right, top to bottom).
 * 
 * Deposit grid position numbering:
 *   1  2  3  4  5  6  7  8
 *   9 10 11 12 13 14 15 16
 *  17 18 19 20 21 22 23 24
 *  25 26 27 28 29 30 31 32
 *  33 34 35 36 37 38 39 40
 * 
 * @param depositPosition The deposit grid position number (1-40, where 1 is top-left)
 * @param firstDepositXSteps The X position (in steps) of deposit grid position 1
 * @param firstDepositYSteps The Y position (in steps) of deposit grid position 1
 * @return true if successful, false if depositPosition is out of range
 */
bool navigateToDepositGridPosition(int depositPosition, long firstDepositXSteps, long firstDepositYSteps) {
  // Validate deposit grid position
  if (depositPosition < 1 || depositPosition > DEPOSIT_GRID_TOTAL_POSITIONS) {
    Serial.print(F("ERROR: Deposit grid position "));
    Serial.print(depositPosition);
    Serial.print(F(" is out of range (1-"));
    Serial.print(DEPOSIT_GRID_TOTAL_POSITIONS);
    Serial.println(F(")"));
    return false;
  }
  
  // Convert 1-indexed position to 0-indexed
  int positionIndex = depositPosition - 1;
  
  // Calculate row and column (0-indexed)
  int column = positionIndex % DEPOSIT_GRID_COLUMNS;  // 0-7
  int row = positionIndex / DEPOSIT_GRID_COLUMNS;     // 0-4
  
  // Calculate physical offsets from first deposit position
  float xOffsetMm = column * DEPOSIT_GRID_COL_SPACING_MM;
  float yOffsetMm = row * DEPOSIT_GRID_ROW_SPACING_MM;
  
  // Convert to steps
  long xSteps = firstDepositXSteps + distanceToSteps(xOffsetMm);
  long ySteps = firstDepositYSteps + distanceToSteps(yOffsetMm);
  
  // Debug output
  Serial.print(F("Navigating to deposit grid position "));
  Serial.print(depositPosition);
  Serial.print(F(" (Row "));
  Serial.print(row + 1);
  Serial.print(F(", Column "));
  Serial.print(column + 1);
  Serial.print(F(") -> X: "));
  Serial.print(xSteps);
  Serial.print(F(" steps, Y: "));
  Serial.print(ySteps);
  Serial.println(F(" steps"));
  
  // Move to the calculated position
  moveToXYPosition(xSteps, ySteps);
  
  return true;
}

/**
 * Navigates through a sequence of grid positions from startPosition to endPosition (inclusive).
 * Useful for processing multiple samples in sequence.
 * 
 * @param startPosition Starting grid position (1-40)
 * @param endPosition Ending grid position (1-40, must be >= startPosition)
 * @param firstPositionXSteps The X position (in steps) of grid position 1
 * @param firstPositionYSteps The Y position (in steps) of grid position 1
 * @return true if successful, false if parameters are invalid
 */
bool navigateThroughGridPositions(int startPosition, int endPosition, 
                                   long firstPositionXSteps, long firstPositionYSteps) {
  // Validate parameters
  if (startPosition < 1 || startPosition > GRID_TOTAL_POSITIONS) {
    Serial.print(F("ERROR: Start position "));
    Serial.print(startPosition);
    Serial.println(F(" is out of range"));
    return false;
  }
  
  if (endPosition < 1 || endPosition > GRID_TOTAL_POSITIONS) {
    Serial.print(F("ERROR: End position "));
    Serial.print(endPosition);
    Serial.println(F(" is out of range"));
    return false;
  }
  
  if (endPosition < startPosition) {
    Serial.println(F("ERROR: End position must be >= start position"));
    return false;
  }
  
  Serial.print(F("Navigating through grid positions "));
  Serial.print(startPosition);
  Serial.print(F(" to "));
  Serial.println(endPosition);
  
  // Navigate to each position in sequence
  for (int pos = startPosition; pos <= endPosition; pos++) {
    if (!navigateToGridPosition(pos, firstPositionXSteps, firstPositionYSteps)) {
      return false;
    }
    // Optional: Add a small delay between positions if needed
    // delay(100);
  }
  
  Serial.println(F("Grid navigation sequence complete."));
  return true;
}

// ---------------------- Utility Prototypes ----------------------
long distanceToSteps(float distanceMm);
bool navigateToGridPosition(int gridPosition, long firstPositionXSteps, long firstPositionYSteps);
bool navigateToDepositGridPosition(int depositPosition, long firstDepositXSteps, long firstDepositYSteps);
bool navigateThroughGridPositions(int startPosition, int endPosition, 
                                   long firstPositionXSteps, long firstPositionYSteps);
void enableDriverPin(int pin, bool enableActiveLow);
bool isLimitPressed(uint8_t pin);
bool checkMaxLimits();
bool checkAllLimits();
bool areDoorsClosed();
void waitForDoorsToClose();
bool requestImageCapture();
bool waitForPythonResponse(const String &expectedResponse, unsigned long timeoutMs);
bool checkPythonReady();
void setXYPosition(long xSteps, long ySteps);
void moveCoreXYRelative(long deltaX, long deltaY, float maxSpeed, float accel, bool checkLimits = true);
void moveToXYPosition(long xSteps, long ySteps);
void calibrationStep();
long requestTotalSamples();
long requestSignedLong(const __FlashStringHelper *prompt);
bool parseSignedLong(const String &text, long &value);
void runSamplingSequence();
void waitForHostCommand(const String &expected);
void sendToolheadCommand(const String &cmd);
bool waitForToolheadResponse(const String &expectedResponse, unsigned long timeoutMs = TOOLHEAD_RESPONSE_TIMEOUT_MS);
bool requestSampleFromToolhead();
bool requestDepositFromToolhead();

// ---------------------- Setup ----------------------
void setup() {
  Serial.begin(115200);
  Serial.setTimeout(HOST_SERIAL_TIMEOUT_MS);
  Serial1.begin(TOOLHEAD_SERIAL_BAUD);
  Serial1.setTimeout(TOOLHEAD_SERIAL_TIMEOUT_MS);
  while (!Serial) {
    ; // Wait for serial monitor on boards that need it (harmless on Mega)
  }
  Serial.println();
  Serial.println(F("Desktop Soil Sampler - CoreXY Calibration"));
  Serial.println(F("Initializing..."));

  // Configure limit switches with internal pull-ups if they are active LOW.
  pinMode(X_MIN_LIMIT_PIN, LIMIT_ACTIVE_LOW ? INPUT_PULLUP : INPUT);
  pinMode(Y_MIN_LIMIT_PIN, LIMIT_ACTIVE_LOW ? INPUT_PULLUP : INPUT);
  pinMode(X_MAX_LIMIT_PIN, LIMIT_ACTIVE_LOW ? INPUT_PULLUP : INPUT);
  pinMode(Y_MAX_LIMIT_PIN, LIMIT_ACTIVE_LOW ? INPUT_PULLUP : INPUT);
  
  // Configure door limit switches
  pinMode(DOOR_1_LIMIT_PIN, LIMIT_ACTIVE_LOW ? INPUT_PULLUP : INPUT);
  pinMode(DOOR_2_LIMIT_PIN, LIMIT_ACTIVE_LOW ? INPUT_PULLUP : INPUT);

  // Configure (optional) enable pins.
  enableDriverPin(MOTOR_A_ENABLE_PIN, true);
  enableDriverPin(MOTOR_B_ENABLE_PIN, true);

  // Configure steppers for general travel.
  motorA.setMaxSpeed(TRAVEL_MAX_SPEED);
  motorA.setAcceleration(TRAVEL_ACCEL);
  motorB.setMaxSpeed(TRAVEL_MAX_SPEED);
  motorB.setAcceleration(TRAVEL_ACCEL);

  Serial.println(F("Starting Calibration sequence..."));
  //Serial.println(F("Starting homing sequence..."));
  calibrationStep();
  //homeSystem();
  Serial.println(F("Homing complete. System ready at (0, 0)."));
  Serial.println(F("Calibration complete."));

  // Ask operator for how many grid samples to process this run.
  totalSamples = requestTotalSamples();
  firstSampleBaseXSteps = requestSignedLong(F("ENTER FIRST SAMPLE BASE X STEPS:"));
  firstSampleBaseYSteps = requestSignedLong(F("ENTER FIRST SAMPLE BASE Y STEPS:"));
  firstDepositGridXSteps = requestSignedLong(F("ENTER FIRST DEPOSIT GRID X STEPS:"));
  firstDepositGridYSteps = requestSignedLong(F("ENTER FIRST DEPOSIT GRID Y STEPS:"));
  samplingSequenceArmed = true;
  samplingSequenceComplete = false;
  
}

// ---------------------- Main Loop ----------------------
void loop() {
  // Request number of grid positions to navigate (always starts at position 1)
  Serial.println();
  Serial.println(F("=== Grid Navigation ==="));
  Serial.print(F("Enter total number of samples to process (1-"));
  Serial.print(GRID_TOTAL_POSITIONS);
  Serial.print(F(") - starting from position 1: "));
  
  // Wait for user input
  while (!Serial.available()) {
    delay(10);
  }
  
  String input = Serial.readStringUntil('\n');
  input.trim();
  
  int numPositions = input.toInt();
  
  // Validate input
  if (numPositions < 1 || numPositions > GRID_TOTAL_POSITIONS) {
    Serial.print(F("ERROR: Invalid input. Please enter a number between 1 and "));
    Serial.println(GRID_TOTAL_POSITIONS);
    return;
  }
  
  Serial.print(F("Navigating through positions 1 to "));
  Serial.print(numPositions);
  Serial.print(F(" ("));
  Serial.print(numPositions);
  Serial.println(F(" total samples)..."));
  Serial.println();
  
  // Check for limit switches before starting
  if (checkAllLimits()) {
    Serial.println(F("ERROR: Limit switch triggered before navigation. Aborting."));
    return;
  }
  
  // Check doors before starting - wait if open
  if (!areDoorsClosed()) {
    waitForDoorsToClose();
  }
  
  // Optional: Check if Python program is ready (non-blocking - continues if no response)
  Serial.println(F("Checking Python connection (optional)..."));
  if (Serial.available() > 0) {
    // Clear any existing data
    while (Serial.available() > 0) {
      Serial.read();
    }
  }
  // Note: Python ready check is optional - system will continue even if Python is not connected
  
  // Navigate through each position with 3 second delay
  for (int pos = 1; pos <= numPositions; pos++) {
    // Check for limit switches before each position
    if (checkAllLimits()) {
      Serial.println(F("ERROR: Limit switch triggered during navigation. Terminating."));
      return;
    }
    
    // Check doors before each position - wait if open
    if (!areDoorsClosed()) {
      waitForDoorsToClose();
    }
    
    Serial.print(F("Position "));
    Serial.print(pos);
    Serial.print(F(" of "));
    Serial.println(numPositions);
    
    // Navigate to the grid position
    if (!navigateToGridPosition(pos, firstSampleBaseXSteps, firstSampleBaseYSteps)) {
      Serial.println(F("ERROR: Failed to navigate to position. Aborting."));
      return;
    }
    
    // Check for limit switches after movement
    if (checkAllLimits()) {
      Serial.println(F("ERROR: Limit switch triggered after movement. Terminating."));
      return;
    }
    
    // Check doors after movement - wait if open
    if (!areDoorsClosed()) {
      waitForDoorsToClose();
    }
    
    // Request image capture from Python program
    Serial.println(F("Requesting image capture..."));
    if (!requestImageCapture()) {
      Serial.println(F("WARNING: Image capture failed, but continuing to next position."));
      // Continue even if image capture fails - don't abort the sequence
    }
    
    // Store current position before shifting (for returning after deposit)
    long originalXSteps = currentX();
    long originalYSteps = currentY();
    
    // Shift Y-axis to make room for toolhead
    Serial.print(F("Shifting Y-axis by "));
    Serial.print(Y_OFFSET_FOR_TOOLHEAD_MM);
    Serial.println(F(" mm to make room for toolhead..."));
    long yOffsetSteps = distanceToSteps(Y_OFFSET_FOR_TOOLHEAD_MM);
    moveCoreXYRelative(0L, yOffsetSteps, TRAVEL_MAX_SPEED, TRAVEL_ACCEL);
    
    // Check for limit switches after Y shift
    if (checkAllLimits()) {
      Serial.println(F("ERROR: Limit switch triggered after Y shift. Terminating."));
      return;
    }
    
    // Check doors after Y shift - wait if open
    if (!areDoorsClosed()) {
      waitForDoorsToClose();
    }
    
    // Request sample from toolhead Arduino
    Serial.println(F("Requesting sample from toolhead..."));
    if (!requestSampleFromToolhead()) {
      Serial.println(F("ERROR: Sample taking failed. Aborting sequence."));
      return;
    }
    
    // Check for limit switches after sampling
    if (checkAllLimits()) {
      Serial.println(F("ERROR: Limit switch triggered after sampling. Terminating."));
      return;
    }
    
    // Check doors after sampling - wait if open
    if (!areDoorsClosed()) {
      waitForDoorsToClose();
    }
    
    // Navigate to deposit grid position (same position number as sample grid)
    // Note: We're still at the shifted Y position, will navigate directly to deposit grid
    Serial.println(F("Navigating to deposit grid position..."));
    if (!navigateToDepositGridPosition(pos, firstDepositGridXSteps, firstDepositGridYSteps)) {
      Serial.println(F("ERROR: Failed to navigate to deposit grid position. Aborting."));
      return;
    }
    
    // Check for limit switches after movement to deposit grid
    if (checkAllLimits()) {
      Serial.println(F("ERROR: Limit switch triggered after movement to deposit grid. Terminating."));
      return;
    }
    
    // Check doors after movement to deposit grid - wait if open
    if (!areDoorsClosed()) {
      waitForDoorsToClose();
    }
    
    // Request deposit from toolhead Arduino
    Serial.println(F("Requesting sample deposit from toolhead..."));
    if (!requestDepositFromToolhead()) {
      Serial.println(F("ERROR: Sample deposit failed. Aborting sequence."));
      return;
    }
    
    // Check for limit switches after deposit
    if (checkAllLimits()) {
      Serial.println(F("ERROR: Limit switch triggered after deposit. Terminating."));
      return;
    }
    
    // Check doors after deposit - wait if open
    if (!areDoorsClosed()) {
      waitForDoorsToClose();
    }
    
    // Return to original sample grid position to redeposit any excess
    Serial.println(F("Returning to original sample grid position to redeposit excess..."));
    moveToXYPosition(originalXSteps, originalYSteps);
    
    // Check for limit switches after returning to original position
    if (checkAllLimits()) {
      Serial.println(F("ERROR: Limit switch triggered after returning to original position. Terminating."));
      return;
    }
    
    // Check doors after returning - wait if open
    if (!areDoorsClosed()) {
      waitForDoorsToClose();
    }
    
    Serial.println();
  }
  
  Serial.println(F("Grid navigation complete!"));
  Serial.println(F("Ready for next command."));
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

bool checkAllLimits() {
  // Returns true if any limit switch (min or max) is triggered
  if (isLimitPressed(X_MIN_LIMIT_PIN)) {
    Serial.println(F("WARNING: X MIN limit triggered!"));
    return true;
  }
  if (isLimitPressed(Y_MIN_LIMIT_PIN)) {
    Serial.println(F("WARNING: Y MIN limit triggered!"));
    return true;
  }
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

bool areDoorsClosed() {
  // Returns true if both doors are closed (limit switches not triggered)
  // Returns false if either door is open (limit switch triggered)
  bool door1Closed = !isLimitPressed(DOOR_1_LIMIT_PIN);
  bool door2Closed = !isLimitPressed(DOOR_2_LIMIT_PIN);
  
  if (!door1Closed) {
    Serial.println(F("WARNING: Door 1 is open!"));
  }
  if (!door2Closed) {
    Serial.println(F("WARNING: Door 2 is open!"));
  }
  
  return door1Closed && door2Closed;
}

void waitForDoorsToClose() {
  // Pauses execution until both doors are closed
  while (!areDoorsClosed()) {
    Serial.println(F("PAUSED: Waiting for doors to close..."));
    delay(500);  // Check every 500ms
  }
  Serial.println(F("Doors closed. Resuming operation."));
}

// ---------------------- Python Communication Functions ----------------------
bool waitForPythonResponse(const String &expectedResponse, unsigned long timeoutMs) {
  // Waits for a specific response from Python program via Serial
  // Returns true if expected response received, false if timeout
  unsigned long startTime = millis();
  String receivedResponse = "";
  
  Serial.print(F("Waiting for Python response: "));
  Serial.print(expectedResponse);
  Serial.print(F(" (timeout: "));
  Serial.print(timeoutMs);
  Serial.println(F(" ms)"));
  
  while (millis() - startTime < timeoutMs) {
    if (Serial.available() > 0) {
      receivedResponse = Serial.readStringUntil('\n');
      receivedResponse.trim();
      
      Serial.print(F("Received: "));
      Serial.println(receivedResponse);
      
      if (receivedResponse == expectedResponse) {
        Serial.println(F("Python response received successfully."));
        return true;
      } else {
        Serial.print(F("Unexpected response. Expected: "));
        Serial.print(expectedResponse);
        Serial.print(F(", Got: "));
        Serial.println(receivedResponse);
      }
    }
    delay(10);  // Small delay to prevent excessive CPU usage
  }
  
  Serial.print(F("ERROR: Timeout waiting for Python response: "));
  Serial.println(expectedResponse);
  return false;
}

bool checkPythonReady() {
  // Checks if Python program is ready by sending a ready check
  // Python should respond with PYTHON_READY_RESPONSE
  Serial.println(F("Checking if Python program is ready..."));
  Serial.println(PYTHON_READY_RESPONSE);
  
  return waitForPythonResponse(PYTHON_READY_RESPONSE, 5000UL);  // 5 second timeout for ready check
}

bool requestImageCapture() {
  // Sends image capture command to Python and waits for confirmation
  // Returns true if image capture confirmed, false if timeout or error
  Serial.println(F("Requesting image capture from Python..."));
  
  // Send capture command to Python
  Serial.println(PYTHON_CAPTURE_COMMAND);
  
  // Wait for Python to confirm image is captured
  bool success = waitForPythonResponse(PYTHON_CAPTURE_RESPONSE, PYTHON_RESPONSE_TIMEOUT_MS);
  
  if (success) {
    Serial.println(F("Image capture completed successfully."));
  } else {
    Serial.println(F("ERROR: Image capture failed or timed out."));
  }
  
  return success;
}

// ---------------------- Toolhead Communication Functions ----------------------
void sendToolheadCommand(const String &cmd) {
  // Sends a command to the toolhead Arduino via Serial1
  Serial.print(F("Sending to toolhead: "));
  Serial.println(cmd);
  Serial1.println(cmd);
}

bool waitForToolheadResponse(const String &expectedResponse, unsigned long timeoutMs) {
  // Waits for a specific response from toolhead Arduino via Serial1
  // Returns true if expected response received, false if timeout
  unsigned long startTime = millis();
  String receivedResponse = "";
  
  Serial.print(F("Waiting for toolhead response: "));
  Serial.print(expectedResponse);
  Serial.print(F(" (timeout: "));
  Serial.print(timeoutMs);
  Serial.println(F(" ms)"));
  
  while (millis() - startTime < timeoutMs) {
    if (Serial1.available() > 0) {
      receivedResponse = Serial1.readStringUntil('\n');
      receivedResponse.trim();
      
      Serial.print(F("Toolhead received: "));
      Serial.println(receivedResponse);
      
      if (receivedResponse == expectedResponse) {
        Serial.println(F("Toolhead response received successfully."));
        return true;
      } else {
        Serial.print(F("Unexpected toolhead response. Expected: "));
        Serial.print(expectedResponse);
        Serial.print(F(", Got: "));
        Serial.println(receivedResponse);
      }
    }
    delay(10);  // Small delay to prevent excessive CPU usage
  }
  
  Serial.print(F("ERROR: Timeout waiting for toolhead response: "));
  Serial.println(expectedResponse);
  return false;
}

bool requestSampleFromToolhead() {
  // Sends sample command to toolhead and waits for confirmation
  // Returns true if sample taken confirmed, false if timeout or error
  Serial.println(F("Requesting sample from toolhead..."));
  
  // Send sample command to toolhead
  sendToolheadCommand(TOOLHEAD_SAMPLE_COMMAND);
  
  // Wait for toolhead to confirm sample is taken
  bool success = waitForToolheadResponse(TOOLHEAD_SAMPLE_RESPONSE, TOOLHEAD_RESPONSE_TIMEOUT_MS);
  
  if (success) {
    Serial.println(F("Sample taken successfully."));
  } else {
    Serial.println(F("ERROR: Sample taking failed or timed out."));
  }
  
  return success;
}

bool requestDepositFromToolhead() {
  // Sends deposit command to toolhead and waits for confirmation
  // Returns true if deposit confirmed, false if timeout or error
  Serial.println(F("Requesting sample deposit from toolhead..."));
  
  // Send deposit command to toolhead
  sendToolheadCommand(TOOLHEAD_DEPOSIT_COMMAND);
  
  // Wait for toolhead to confirm sample is deposited
  bool success = waitForToolheadResponse(TOOLHEAD_DEPOSIT_RESPONSE, TOOLHEAD_RESPONSE_TIMEOUT_MS);
  
  if (success) {
    Serial.println(F("Sample deposited successfully."));
  } else {
    Serial.println(F("ERROR: Sample deposit failed or timed out."));
  }
  
  return success;
}

void setXYPosition(long xSteps, long ySteps) {
  motorA.setCurrentPosition(coreXY_A(xSteps, ySteps));
  motorB.setCurrentPosition(coreXY_B(xSteps, ySteps));
}

// Simple wrapper that converts requested absolute XY to the relative planner.
void moveToXYPosition(long xSteps, long ySteps) {
  const long deltaX = xSteps - currentX();
  const long deltaY = ySteps - currentY();
  moveCoreXYRelative(deltaX, deltaY, TRAVEL_MAX_SPEED, TRAVEL_ACCEL);
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

void calibrationStep() {

  //Y MIN CALIBRATION
  Serial.println(F(" -> Starting Y-axis homing"));
  
  //Checks if limit is already pressed
  if (isLimitPressed(Y_MIN_LIMIT_PIN)) {
    Serial.println(F("    Y limit already active. Backing off before homing."));
    moveCoreXYRelative(0L, BACKOFF_STEPS, BACKOFF_SPEED, BACKOFF_ACCEL);
  }

  //Moves toward the minimum endstop (towrads interface, long side)
  motorA.setAcceleration(HOMING_ACCEL);
  motorB.setAcceleration(HOMING_ACCEL);
  motorA.setMaxSpeed(absf(HOMING_SPEED));
  motorB.setMaxSpeed(absf(HOMING_SPEED));
  motorA.setSpeed(-HOMING_SPEED);
  motorB.setSpeed(HOMING_SPEED);
  Serial.println(F(" Seeking Y minimum limit..."));

  //While limit switch is not pressed, keep moving
  while (!isLimitPressed(Y_MIN_LIMIT_PIN)) {
    motorA.runSpeed();
    motorB.runSpeed();
  }
  motorA.setSpeed(0.0f);
  motorB.setSpeed(0.0f);
  Serial.println(F("    Y limit triggered."));
  delay(50); // Allow the switch to settle.

  //Switch hit backing off
  Serial.println(F("    Backing off Y axis to clear the switch."));
  moveCoreXYRelative(0L, BACKOFF_STEPS, BACKOFF_SPEED, BACKOFF_ACCEL);

  //Recording position
  setXYPosition(currentX(), 0L);
  Serial.println(F("    Y axis zero set."));



  //X MIN CALIBRATION (towards interface, short side)
  Serial.println(F(" -> Starting X-axis homing"));
  Serial.println(F("    Seeking X minimum limit..."));

  //Checks if limit is already pressed
  if (isLimitPressed(X_MIN_LIMIT_PIN)) {
    Serial.println(F("    X limit already active. Backing off before homing."));
    moveCoreXYRelative(BACKOFF_STEPS, 0L, BACKOFF_SPEED, BACKOFF_ACCEL);
  }
  //Moves toward the minimum endstop
  motorA.setAcceleration(HOMING_ACCEL);
  motorB.setAcceleration(HOMING_ACCEL);
  motorA.setMaxSpeed(absf(HOMING_SPEED));
  motorB.setMaxSpeed(absf(HOMING_SPEED));
  motorA.setSpeed(HOMING_SPEED);
  motorB.setSpeed(HOMING_SPEED);
  
  //While limit switch is not pressed, keep moving
  while (!isLimitPressed(X_MIN_LIMIT_PIN)) {
    motorA.runSpeed();
    motorB.runSpeed();
  }
  motorA.setSpeed(0.0f);
  motorB.setSpeed(0.0f);
  Serial.println(F("    X limit triggered."));
  delay(50); // Allow the switch to settle.
  Serial.println(F("  Backing off X axis to clear the switch."));  
  moveCoreXYRelative(-BACKOFF_STEPS, 0L, BACKOFF_SPEED, BACKOFF_ACCEL);
  //Recording position
  setXYPosition(0L, currentY());
  Serial.println(F("    X axis zero set."));



  //X MAX CALIBRATION
  Serial.println(F(" -> Starting X-axis maximum calibration"));
  motorA.setAcceleration(HOMING_ACCEL);
  motorB.setAcceleration(HOMING_ACCEL);
  motorA.setMaxSpeed(absf(HOMING_SPEED));
  motorB.setMaxSpeed(absf(HOMING_SPEED));
  motorA.setSpeed(-HOMING_SPEED);
  motorB.setSpeed(-HOMING_SPEED);

  if (isLimitPressed(X_MAX_LIMIT_PIN)) {
    Serial.println(F("    X MAX limit already active. Backing off before search."));
    moveCoreXYRelative(-BACKOFF_STEPS, 0L, BACKOFF_SPEED, BACKOFF_ACCEL);
  }

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
  moveCoreXYRelative(BACKOFF_STEPS, 0L, BACKOFF_SPEED, BACKOFF_ACCEL, false);



  //Y MAX CALIBRATION
  Serial.println(F(" -> Starting Y-axis maximum calibration"));
  motorA.setAcceleration(HOMING_ACCEL);
  motorB.setAcceleration(HOMING_ACCEL);
  motorA.setMaxSpeed(absf(HOMING_SPEED));
  motorB.setMaxSpeed(absf(HOMING_SPEED));
  motorA.setSpeed(HOMING_SPEED);
  motorB.setSpeed(-HOMING_SPEED);
  Serial.println(F("    Seeking Y maximum limit..."));
  while (!isLimitPressed(Y_MAX_LIMIT_PIN)) {
    motorA.runSpeed();
    motorB.runSpeed();
  }
  motorA.setSpeed(0.0f);
  motorB.setSpeed(0.0f);
  Serial.println(F("    Y MAX limit triggered."));
  delay(50); // Allow the switch to settle.

  // Record the maximum Y position
  maxYSteps = currentY();
  Serial.print(F("    Y maximum position: "));
  Serial.print(maxYSteps);
  Serial.println(F(" steps"));
  Serial.println(F("    Backing off Y axis to clear the switch."));
  moveCoreXYRelative(0L, -BACKOFF_STEPS, BACKOFF_SPEED, BACKOFF_ACCEL, false); 

  Serial.println(F(" -> Returning to home position (0, 0)"));
  Serial.print(F("    Current position before return: X="));
  Serial.print(currentX());
  Serial.print(F(", Y="));
  Serial.println(currentY());
  moveCoreXYRelative(0L, -currentY(), TRAVEL_MAX_SPEED, TRAVEL_ACCEL);
  moveCoreXYRelative(-currentX(), 0L, TRAVEL_MAX_SPEED, TRAVEL_ACCEL);
  Serial.print(F("    Position after return: X="));
  Serial.print(currentX());
  Serial.print(F(", Y="));
  Serial.println(currentY());
  Serial.println(F(" -> Gantry reset to home."));

}

// ---------------------- Input Functions ----------------------
long requestTotalSamples() {
  Serial.print(F("Enter total number of samples (1-"));
  Serial.print(GRID_TOTAL_POSITIONS);
  Serial.print(F("): "));
  
  while (!Serial.available()) {
    delay(10);
  }
  
  String input = Serial.readStringUntil('\n');
  input.trim();
  long value = input.toInt();
  
  if (value < 1 || value > GRID_TOTAL_POSITIONS) {
    Serial.print(F("Invalid input. Using default: 1"));
    Serial.println();
    return 1;
  }
  
  Serial.println(value);
  return value;
}

long requestSignedLong(const __FlashStringHelper *prompt) {
  Serial.print(prompt);
  
  while (!Serial.available()) {
    delay(10);
  }
  
  String input = Serial.readStringUntil('\n');
  input.trim();
  
  long value;
  if (parseSignedLong(input, value)) {
    Serial.println(value);
    return value;
  } else {
    Serial.println(F("Invalid input. Using default: 0"));
    return 0;
  }
}

bool parseSignedLong(const String &text, long &value) {
  if (text.length() == 0) {
    return false;
  }
  
  char *endptr;
  value = strtol(text.c_str(), &endptr, 10);
  
  // Check if entire string was parsed
  return (*endptr == '\0');
}


