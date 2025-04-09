#include <BluetoothSerial.h> // ESP32 Bluetooth Library

// Create Bluetooth object
BluetoothSerial SerialBT;

#include <TB6612_ESP32.h>

// Motor 1 (Left Motor)
int motor1Pin1 = 17;  
int motor1Pin2 = 19;  
int enable1Pin = 15;  // PWM pin for speed control

// Motor 2 (Right Motor)
int motor2Pin1 = 26;  
int motor2Pin2 = 27;  
int enable2Pin = 14; // PWM pin for speed control

int speedValue = 150; // Speed (0-255)

void setup() {
  Serial.begin(115200);
  SerialBT.begin("Motor_car"); // Bluetooth name
  Serial.println("Bluetooth Started! Connect via Serial Bluetooth Terminal");

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
  if (SerialBT.available()) {  // Check if Bluetooth command is received
    char command = SerialBT.read();
    Serial.print("Received: ");
    Serial.println(command);

    switch (command) {
      case 'F': // Move Forward
        Serial.println("Moving Forward");
        moveForward();
        break;

      case 'B': // Move Backward
        Serial.println("Moving Backward");
        moveBackward();
        break;

      case 'L': // Turn Left
        Serial.println("Turning Left");
        turnLeft();
        break;

      case 'R': // Turn Right (Auto-stop after 2 sec)
        Serial.println("Turning Right for 2 sec...");
        turnRight();
        delay(2000); // Wait 2 seconds
        stopMotors(); // Stop after turning
        break;

      case 'S': // Stop both motors
        Serial.println("Stopping");
        stopMotors();
        break;

      default:
        Serial.println("Invalid Command! Use F, B, L, R, or S.");
        break;
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
