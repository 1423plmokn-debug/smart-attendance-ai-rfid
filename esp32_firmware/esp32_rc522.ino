/*
 * ==================================================================================
 * Smart Attendance System — RFID RC522 & ESP32-S Firmware
 * الجامعة التقنية الوسطى / كلية البوليتكنك — Middle Technical University
 * 
 * Hardware:
 *   - ESP32-S / ESP32-WROOM-32 Microcontroller
 *   - MFRC522 RFID 13.56 MHz Reader Module
 *   - Onboard Status LED (GPIO 2)
 *
 * Pin Connections (Standard ESP32 VSPI Hardware SPI):
 *   --------------------------------------------------------
 *   RC522 Pin    | ESP32-S GPIO    | Description
 *   -------------+-----------------+-----------------------
 *   3.3V (VCC)   | 3V3             | Power (CRITICAL: 3.3V ONLY, NEVER 5V!)
 *   RST          | GPIO 22         | Reset
 *   GND          | GND             | Common Ground
 *   MISO         | GPIO 19         | SPI Master In Slave Out (VSPI MISO)
 *   MOSI         | GPIO 23         | SPI Master Out Slave In (VSPI MOSI)
 *   SCK          | GPIO 18         | SPI Clock (VSPI SCK)
 *   SDA (SS)     | GPIO 5          | SPI Chip Select / Slave Select (VSPI SS)
 *   --------------------------------------------------------
 *
 * Serial Communication:
 *   - Baud Rate: 115200 bps
 *   - Output Format: Strictly "UID:XX:XX:XX:XX" (Uppercase 2-digit Hex bytes)
 *   - Example: UID:3A:5B:92:F1
 *   - Hardware Port: COM5 (or assigned USB Serial Port)
 *
 * Built-in Features:
 *   - Non-blocking millisecond debounce cooldown (prevents duplicate serial spam)
 *   - Hardware LED confirmation pulse upon card detection
 *   - Auto PICC halt and crypto stop to avoid card lockups
 * ==================================================================================
 */

#include <SPI.h>
#include <MFRC522.h>

// ----------------------------------------------------------------------------------
// Pin Configurations
// ----------------------------------------------------------------------------------
#define SS_PIN          5     // RC522 SDA / SS (Slave Select)
#define RST_PIN         22    // RC522 RST (Reset)
#define STATUS_LED_PIN  2     // Onboard Blue LED on ESP32 (GPIO 2)

// ----------------------------------------------------------------------------------
// Timing & Cooldown Constants (Non-blocking)
// ----------------------------------------------------------------------------------
const unsigned long SCAN_COOLDOWN_MS = 1500;  // 1.5 seconds cooldown between identical card scans
const unsigned long LED_PULSE_MS     = 100;   // Visual confirmation blip duration

// ----------------------------------------------------------------------------------
// Global Objects & State Variables
// ----------------------------------------------------------------------------------
MFRC522 rfid(SS_PIN, RST_PIN);

unsigned long lastScanTime = 0;
String lastScannedUID = "";
unsigned long ledTurnOffTime = 0;
bool ledActive = false;

// ----------------------------------------------------------------------------------
// Setup
// ----------------------------------------------------------------------------------
void setup() {
  // Initialize Serial interface at high-speed 115200 baud
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    ; // Wait briefly for USB Serial interface to enumerate
  }

  // Initialize LED pin for visual feedback
  pinMode(STATUS_LED_PIN, OUTPUT);
  digitalWrite(STATUS_LED_PIN, LOW);

  // Initialize SPI bus with standard ESP32 VSPI pins (SCK: 18, MISO: 19, MOSI: 23, SS: 5)
  SPI.begin(18, 19, 23, 5);

  // Initialize MFRC522 RFID reader
  rfid.PCD_Init();
  delay(10); // Settle time

  // Read & verify RFID reader firmware version to ensure wiring is correct
  byte version = rfid.PCD_ReadRegister(rfid.VersionReg);
  if (version == 0x00 || version == 0xFF) {
    Serial.println("STATUS:ERROR_RC522_NOT_DETECTED - Check wiring to GPIO 5, 18, 19, 22, 23");
  } else {
    Serial.println("STATUS:RFID_INITIALIZED_READY");
  }

  // Quick dual-blink to show hardware startup complete
  digitalWrite(STATUS_LED_PIN, HIGH);
  delay(100);
  digitalWrite(STATUS_LED_PIN, LOW);
  delay(100);
  digitalWrite(STATUS_LED_PIN, HIGH);
  delay(100);
  digitalWrite(STATUS_LED_PIN, LOW);
}

// ----------------------------------------------------------------------------------
// Main Loop (100% Non-blocking)
// ----------------------------------------------------------------------------------
void loop() {
  // Non-blocking timer to turn off status LED after pulse
  if (ledActive && (millis() >= ledTurnOffTime)) {
    digitalWrite(STATUS_LED_PIN, LOW);
    ledActive = false;
  }

  // Check if a new RFID card/tag is present on the RF antenna
  if (!rfid.PICC_IsNewCardPresent()) {
    return;
  }

  // Read card serial number
  if (!rfid.PICC_ReadCardSerial()) {
    return;
  }

  // Format detected UID strictly as "UID:XX:XX:XX:XX"
  String currentUID = "UID:";
  for (byte i = 0; i < rfid.uid.size; i++) {
    if (i > 0) {
      currentUID += ":";
    }
    if (rfid.uid.uidByte[i] < 0x10) {
      currentUID += "0";
    }
    currentUID += String(rfid.uid.uidByte[i], HEX);
  }
  currentUID.toUpperCase();

  unsigned long now = millis();

  // Cooldown filter: Prevent duplicate serial spam if card lingers on reader
  if ((currentUID == lastScannedUID) && (now - lastScanTime < SCAN_COOLDOWN_MS)) {
    // Card is still resting on the sensor within the cooldown window — ignore
  } else {
    // Valid new card scan! Send payload strictly over Serial to COM5
    Serial.println(currentUID);
    lastScannedUID = currentUID;
    lastScanTime = now;

    // Trigger visual LED pulse
    digitalWrite(STATUS_LED_PIN, HIGH);
    ledTurnOffTime = now + LED_PULSE_MS;
    ledActive = true;
  }

  // Halt PICC and stop encryption on PCD to prepare for subsequent scans cleanly
  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();
}
