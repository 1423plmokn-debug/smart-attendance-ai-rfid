# ESP32-S + RC522 RFID Attendance Firmware
**الجامعة التقنية الوسطى / كلية البوليتكنك**
**Middle Technical University / Polytechnic College**

---

### Hardware Wiring Diagram (ESP32-S VSPI)

Connect the **MFRC522** module to the **ESP32-S / ESP32-WROOM-32** as follows:

| RC522 RFID Pin | ESP32-S Pin | Description | Notes |
|:--------------:|:-----------:|:------------|:------|
| **3.3V (VCC)** | **3V3**     | 3.3V Power  | ⚠️ **CRITICAL: NEVER connect to 5V!** The RC522 chip is 3.3V logic only. |
| **RST**        | **GPIO 22** | Hardware Reset | Configurable in code (`RST_PIN`) |
| **GND**        | **GND**     | Ground | Common system ground |
| **MISO**       | **GPIO 19** | SPI Master In / Slave Out | Hardware VSPI MISO |
| **MOSI**       | **GPIO 23** | SPI Master Out / Slave In | Hardware VSPI MOSI |
| **SCK**        | **GPIO 18** | SPI Serial Clock | Hardware VSPI SCK |
| **SDA / SS**   | **GPIO 5**  | SPI Chip Select (CS) | Configurable in code (`SS_PIN`) |
| **IRQ**        | *Not connected* | Interrupt pin | Not required for polling mode |

---

### Flashing with Arduino IDE

1. **Install ESP32 Board Support** (if not already installed):
   - In Arduino IDE, open **File > Preferences**.
   - In *Additional Board Manager URLs*, add:
     `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Open **Tools > Board > Boards Manager**, search for `esp32` by Espressif Systems, and click **Install**.

2. **Install MFRC522 Library**:
   - Open **Tools > Manage Libraries...**
   - Search for `MFRC522` (by GithubCommunity / Miguel Balboa).
   - Click **Install**.

3. **Select Board & Port**:
   - **Board**: `ESP32 Dev Module` (or `DOIT ESP32 DEVKIT V1` / `NodeMCU-32S`).
   - **Upload Speed**: `921600` or `115200`.
   - **Port**: `COM5` (matches the Python backend setting).

4. **Upload the Sketch**:
   - Click **Upload** (Arrow icon).
   - If prompted or if upload is waiting on `Connecting........_____.....`, press and hold the **BOOT** button on the ESP32 until writing starts.

5. **Serial Verification**:
   - Open **Serial Monitor** at **115200 baud**.
   - Tap any RFID card or keyfob onto the reader.
   - You will see the uppercase payload:
     ```text
     STATUS:RFID_INITIALIZED_READY
     UID:3A:5B:92:F1
     ```
