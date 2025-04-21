
// Motor 1 (Left Motor)
int motor1Pin1 = 18;  
int motor1Pin2 = 19;  
int enable1Pin = 5;  // PWM pin for speed control

// Motor 2 (Right Motor)
int motor2Pin1 = 2;  
int motor2Pin2 = 4;  
int enable2Pin = 15; // PWM pin for speed control

int speedValue = 150; // Speed (0-255)

void setup() {
  Serial.begin(115200);


  // Set motor pins as OUTPUT
  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(enable1Pin, OUTPUT);
  
  pinMode(motor2Pin1, OUTPUT);
  pinMode(motor2Pin2, OUTPUT);
  pinMode(enable2Pin, OUTPUT);

  // Set initial motor speed
  analogWrite(enable1Pin, speedValue);
  analogWrite(enable2Pin, speedValue);
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();  // Remove any trailing newline or spaces
    if (input.length() > 0) {
      char command = input.charAt(0);

      switch (command) {
        case 'f': // Move Forward
          Serial.println("Moving Forward");
          moveForward();
          break;
  
        case 'b': // Move Backward
          Serial.println("Moving Backward");
          moveBackward();
          break;
  
        case 'l': // Turn Left
          Serial.println("Turning Left");
          turnLeft();
          delay(500); // Wait 2 seconds
          stopMotors(); // Stop after turning
          moveForward();
          break;
  
        case 'r': // Turn Right (Auto-stop after 2 sec)
          Serial.println("Turning Right for 2 sec...");
          turnRight();
          delay(500); // Wait 2 seconds
          stopMotors(); // Stop after turning
          moveForward();
          break;
  
        case 's': // Stop both motors
          Serial.println("Stopping");
          stopMotors();
          break;
  
        default:
          Serial.println("Invalid Command! Use f, b, l, r, or s.");
          break;
      }
    }
  }
} 
  // Function to move forward
void moveForward() {
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);
}

// Function to move backward
void moveBackward() {
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, HIGH);
}

// Function to turn right
void turnRight() {
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH); // Left motor moves backward
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);  // Right motor moves forward
  analogWrite(enable1Pin, 255);
  analogWrite(enable2Pin, 150); 
}

// Function to turn left (then auto-stop)
void turnLeft() {
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);  // Left motor moves forward
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, HIGH); // Right motor moves backward
  analogWrite(enable1Pin, 150);
  analogWrite(enable2Pin, 255); 
}

// Function to stop all motors
void stopMotors() {
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, LOW);
}
