#include <WiFi.h>
#include <HTTPClient.h>
#include <Preferences.h>
#include <algorithm>

const char* TARGET_SSID_PREFIX = "iPhone";
const char* WIFI_PASS          = "2000nhad";
const char* SERVER_URL         = "http://172.20.10.4:8000/api/v1/esp32/push";
const char* DEVICE_ID          = "ESP32_SOIL_NODE_01";

const int SENSOR_PIN      = 32;
const int NUM_SAMPLES     = 32;
const int TRIM_COUNT      = 4;
const float EMA_ALPHA     = 0.25f;
const unsigned long PUSH_INTERVAL_MS      = 1500;
const unsigned long WIFI_RETRY_INTERVAL_MS = 10000;

const float DEFAULT_DRY_ADC = 2850.0f;
const float DEFAULT_WET_ADC = 1150.0f;

Preferences prefs;
float calDryAdc      = DEFAULT_DRY_ADC;
float calWetAdc      = DEFAULT_WET_ADC;
bool  isCalibrated   = false;
float emaFilteredAdc = -1.0f;

unsigned long lastPushMs      = 0;
unsigned long lastWifiRetryMs = 0;
String resolvedSsid           = "EMAD ...";

struct SoilReading {
  float rawMeanAdc;
  float trimmedAdc;
  float emaAdc;
  float voltageMv;
  float moisturePct;
  const char* statusLabel;
};

SoilReading latestReading = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, "MODERATE"};

void loadCalibration() {
  prefs.begin("ecosensing", true);
  calDryAdc    = prefs.getFloat("dry_adc", DEFAULT_DRY_ADC);
  calWetAdc    = prefs.getFloat("wet_adc", DEFAULT_WET_ADC);
  isCalibrated = prefs.getBool("is_cal", false);
  prefs.end();
}

void saveCalibration(float dryVal, float wetVal, bool calFlag) {
  prefs.begin("ecosensing", false);
  prefs.putFloat("dry_adc", dryVal);
  prefs.putFloat("wet_adc", wetVal);
  prefs.putBool("is_cal", calFlag);
  prefs.end();
  calDryAdc    = dryVal;
  calWetAdc    = wetVal;
  isCalibrated = calFlag;
}

void readFilteredSoil() {
  int rawBuf[NUM_SAMPLES];
  long rawSum = 0;
  long mvSum  = 0;

  for (int i = 0; i < NUM_SAMPLES; i++) {
    rawBuf[i] = analogRead(SENSOR_PIN);
    rawSum += rawBuf[i];
    mvSum  += analogReadMilliVolts(SENSOR_PIN);
    delay(4);
  }

  float rawMean = (float)rawSum / NUM_SAMPLES;
  float mvMean  = (float)mvSum / NUM_SAMPLES;

  std::sort(rawBuf, rawBuf + NUM_SAMPLES);
  long trimmedSum = 0;
  int validCount = NUM_SAMPLES - (2 * TRIM_COUNT);
  for (int i = TRIM_COUNT; i < NUM_SAMPLES - TRIM_COUNT; i++) {
    trimmedSum += rawBuf[i];
  }
  float trimmedMean = (float)trimmedSum / validCount;

  if (emaFilteredAdc < 0.0f) {
    emaFilteredAdc = trimmedMean;
  } else {
    emaFilteredAdc = (EMA_ALPHA * trimmedMean) + ((1.0f - EMA_ALPHA) * emaFilteredAdc);
  }

  float span = calDryAdc - calWetAdc;
  if (span < 100.0f) span = 100.0f;
  float pct = ((calDryAdc - emaFilteredAdc) / span) * 100.0f;
  if (pct < 0.0f)   pct = 0.0f;
  if (pct > 100.0f) pct = 100.0f;

  const char* status = "MODERATE";
  if (pct < 14.0f)       status = "WILTING_CRITICAL";
  else if (pct < 26.0f)  status = "DRY_STRESS";
  else if (pct <= 38.0f) status = "OPTIMAL_FIELD_CAPACITY";
  else                   status = "SATURATED";

  latestReading.rawMeanAdc  = rawMean;
  latestReading.trimmedAdc  = trimmedMean;
  latestReading.emaAdc      = emaFilteredAdc;
  latestReading.voltageMv   = mvMean;
  latestReading.moisturePct = pct;
  latestReading.statusLabel = status;
}

void smartConnectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  WiFi.disconnect(true, true);
  delay(200);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);

  int n = WiFi.scanNetworks(false, true);
  int bestIdx = -1;
  int bestRssi = -1000;

  for (int i = 0; i < n; i++) {
    String s = WiFi.SSID(i);
    if (s.startsWith(TARGET_SSID_PREFIX) || s.indexOf(TARGET_SSID_PREFIX) >= 0) {
      if (WiFi.RSSI(i) > bestRssi) {
        bestRssi = WiFi.RSSI(i);
        bestIdx = i;
      }
    }
  }

  if (bestIdx >= 0) {
    resolvedSsid = WiFi.SSID(bestIdx);
    Serial.printf("[WiFi] Auto-Matched SSID: '%s' (%d dBm, ch=%d)\n",
                  resolvedSsid.c_str(), bestRssi, WiFi.channel(bestIdx));
    WiFi.begin(resolvedSsid.c_str(), WIFI_PASS, WiFi.channel(bestIdx), WiFi.BSSID(bestIdx), true);
  } else {
    Serial.printf("[WiFi] Using direct SSID: '%s'\n", resolvedSsid.c_str());
    WiFi.begin(resolvedSsid.c_str(), WIFI_PASS);
  }

  unsigned long startAttempt = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - startAttempt < 8000)) {
    delay(250);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] CONNECTED! IP: %s | RSSI: %d dBm\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI());
  } else {
    Serial.printf("\n[WiFi] Not connected yet (code=%d) — continuing sensor telemetry.\n", WiFi.status());
  }
  WiFi.scanDelete();
}

void handleSerialCommands() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  cmd.toUpperCase();

  if (cmd == "CAL_DRY") {
    saveCalibration(latestReading.trimmedAdc, calWetAdc, true);
    Serial.printf("[CAL] Saved DRY reference = %.1f ADC\n", calDryAdc);
  } else if (cmd == "CAL_WET") {
    saveCalibration(calDryAdc, latestReading.trimmedAdc, true);
    Serial.printf("[CAL] Saved WET reference = %.1f ADC\n", calWetAdc);
  } else if (cmd == "RESET_CAL") {
    saveCalibration(DEFAULT_DRY_ADC, DEFAULT_WET_ADC, false);
    Serial.println("[CAL] Restored factory calibration defaults.");
  }
}

void setup() {
  Serial.begin(115200);
  delay(400);

  analogReadResolution(12);
  analogSetPinAttenuation(SENSOR_PIN, ADC_11db);

  loadCalibration();
  Serial.println("\n=== EcoSensing AI — Smart Wireless Soil Sentinel ===");
  Serial.printf("[CAL] Dry=%.1f | Wet=%.1f | Status=%s\n",
                calDryAdc, calWetAdc, isCalibrated ? "FIELD_CALIBRATED" : "FACTORY_DEFAULT");

  smartConnectWiFi();
  lastWifiRetryMs = millis();
}

void loop() {
  readFilteredSoil();
  handleSerialCommands();

  if (WiFi.status() != WL_CONNECTED && (millis() - lastWifiRetryMs >= WIFI_RETRY_INTERVAL_MS)) {
    lastWifiRetryMs = millis();
    smartConnectWiFi();
  }

  if (millis() - lastPushMs >= PUSH_INTERVAL_MS) {
    lastPushMs = millis();

    String json = "{";
    json += "\"device_id\":\"" + String(DEVICE_ID) + "\",";
    json += "\"gpio_pin\":32,";
    json += "\"sample_count\":" + String(NUM_SAMPLES) + ",";
    json += "\"raw_adc\":" + String(latestReading.rawMeanAdc, 1) + ",";
    json += "\"filtered_adc\":" + String(latestReading.emaAdc, 1) + ",";
    json += "\"voltage_mv\":" + String(latestReading.voltageMv, 1) + ",";
    json += "\"moisture_pct\":" + String(latestReading.moisturePct, 1) + ",";
    json += "\"soil_status\":\"" + String(latestReading.statusLabel) + "\",";
    json += "\"calibration_status\":\"" + String(isCalibrated ? "FIELD_CALIBRATED" : "FACTORY_DEFAULT") + "\",";
    json += "\"dry_ref_adc\":" + String(calDryAdc, 1) + ",";
    json += "\"wet_ref_adc\":" + String(calWetAdc, 1) + ",";
    json += "\"wifi_rssi\":" + String(WiFi.status() == WL_CONNECTED ? WiFi.RSSI() : 0);
    json += "}";

    Serial.println(json);

    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      http.begin(SERVER_URL);
      http.addHeader("Content-Type", "application/json");
      http.setTimeout(1500);
      int code = http.POST(json);
      if (code > 0) {
        Serial.printf("[HTTP] POST -> %d\n", code);
      }
      http.end();
    }
  }
}