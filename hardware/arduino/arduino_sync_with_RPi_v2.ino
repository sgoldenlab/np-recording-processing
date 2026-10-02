const int LED_INTER_PIN = 13; // LED indicator
const int OUTPUT_PIN = 7;  // output pin to NP DAQ
byte PREVIOUS_BUTTON_STATE = LOW;
const int PING_TIME = 499;  // length of square pulse, in ms

boolean experimentRunning = false;  // running check
boolean pingON = false;
unsigned long startMillis = 0;
unsigned long prevPingMillis = 0;


void setup() {
  pinMode(LED_INTER_PIN, OUTPUT);  // onboard LED, on in between pings
  pinMode(OUTPUT_PIN, OUTPUT);  // output to NP DAQ
  Serial.begin(9600); // Communicate with Raspberry Pi
}

void loop() {   // primarily time-keeping and serial communication here
  unsigned long currentMillis = millis();
  if (!experimentRunning && Serial.available() > 0) {   // start signal received
    char command = Serial.read();
    if (command == 'S') {   // 'S' for Start
      experimentRunning = true;
      startMillis = millis();   // Zero time reference
    } 
  } else if (experimentRunning && Serial.available() > 0) {  // end signal received
    char command = Serial.read();
    if (command == 'E') {   // 'E' for End
      Serial.println(currentMillis - startMillis);  // output timestamp of sync ping
      experimentRunning = false;
    } else if (command == 'R') {  // 'R' for Read -- send timestamp to RPi as needed
      Serial.println(currentMillis - startMillis); 
    }
  }

  if (experimentRunning) {
    if (currentMillis - prevPingMillis > PING_TIME) {
      check_ping();
    }
  }
}

void check_ping() {  // ping to DAQ
  if (!experimentRunning) {
    return;
  }
  unsigned long currentMillis = millis();

  if (currentMillis - prevPingMillis >= PING_TIME) { 
    // save time of ping on / off
    prevPingMillis = currentMillis;
    if (!pingON) {  // ping start
      pingON = true;
      digitalWrite(LED_INTER_PIN, LOW);  // led off - inter-ping end
      // by default sends half-duty cycle (0.5s) ping out every delay time
      digitalWrite(OUTPUT_PIN, HIGH);    // send 1
      Serial.println(currentMillis - startMillis);
    } else if (pingON) {
      pingON = false;
      digitalWrite(OUTPUT_PIN, LOW);     // send 0
      digitalWrite(LED_INTER_PIN, HIGH); // led on - inter-ping start
    }
  }
}

