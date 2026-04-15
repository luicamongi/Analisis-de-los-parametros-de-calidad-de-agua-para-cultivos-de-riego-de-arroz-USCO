#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <math.h>

// ================= WIFI =================
const char* ssid = "iPhone";
const char* password = "Pucheros01";
const char* serverUrl = "http://172.20.10.2:8080/api/mediciones";
const char* UBICACION = "entrada";

// ================= PINES =================
#define I2C_SDA 21
#define I2C_SCL 20

#define ONE_WIRE_BUS 18

#define TRIG_PIN 7
#define ECHO_PIN 8

// ================= OBJETOS =================
Adafruit_ADS1115 ads;
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature ds(&oneWire);

// ================= CALIBRACION =================
// pH: AJUSTA estos dos cuando tomes buffer pH4 y pH7
float PH_V4 = 2.75;   
float PH_V7 = 2.99;   

// EC: ya sacaste el valor de 1413 uS/cm
float EC_V1413 = 2.3984;
float EC_REF = 1413.0;

// ================= CICLO =================
const uint32_t MIN_MS = 60000UL;

const uint32_t T_PH = 0 * MIN_MS;
const uint32_t T_EC = 10 * MIN_MS;
const uint32_t T_TN = 20 * MIN_MS;
const uint32_t CYCLE = 25 * MIN_MS;

uint32_t startCycle = 0;

bool donePH = false;
bool doneEC = false;
bool doneTN = false;

// ================= MODOS =================
// true = solo imprime calibración, no manda HTTP
// false = trabaja normal y envía al backend
bool CAL_MODE = false;

// ================= UTILIDADES =================
float adsToVolts(int16_t adc) {
  return adc * 0.125f / 1000.0f;   // ADS1115
}

// ================= PROMEDIOS =================
// A1 = pH
float leerVoltPHProm() {
  float suma = 0.0;

  for (int i = 0; i < 20; i++) {
    int16_t adc = ads.readADC_SingleEnded(1);
    suma += adsToVolts(adc);
    delay(100);
  }

  return suma / 20.0;
}

// A0 = EC
float leerVoltECProm() {
  float suma = 0.0;

  for (int i = 0; i < 20; i++) {
    int16_t adc = ads.readADC_SingleEnded(0);
    suma += adsToVolts(adc);
    delay(100);
  }

  return suma / 20.0;
}

// ================= CONVERSION pH =================
float convertirPH(float v) {
  float m = (7.0 - 4.0) / (PH_V7 - PH_V4);
  float b = 7.0 - m * PH_V7;
  return (m * v + b);
}

// ================= CONVERSION EC =================
float convertirEC(float v, float temp) {
  float factor = 1.0 + 0.02 * (temp - 25.0);
  float v25 = v / factor;

  float ec = (v25 / EC_V1413) * EC_REF;

  if (ec < 0) ec = 0;
  return ec;
}

// ================= TEMPERATURA =================
float leerTemp() {
  ds.requestTemperatures();
  float t = ds.getTempCByIndex(0);

  if (t == -127.0 || t == 85.0) return NAN;

  return t;
}

// ================= NIVEL =================
float leerNivel() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);

  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);

  if (duration == 0) return NAN;

  float dist = (duration * 0.0343f) / 2.0f;
  return dist;
}

// ================= WIFI =================
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  Serial.print("Conectando WiFi");
  WiFi.begin(ssid, password);

  uint32_t t0 = millis();

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");

    if (millis() - t0 > 20000) {
      Serial.println("\nTimeout WiFi");
      return;
    }
  }

  Serial.println();
  Serial.print("WiFi OK. IP: ");
  Serial.println(WiFi.localIP());
}

// ================= HTTP =================
bool postMedicion(const char* param, float value) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("POST cancelado: sin WiFi");
    return false;
  }

  StaticJsonDocument<128> doc;
  doc["Parametro"] = param;
  doc["Valor"] = value;
  doc["Ubicacion"] = UBICACION;

  String body;
  serializeJson(doc, body);

  HTTPClient http;
  http.setTimeout(5000);
  http.begin(serverUrl);
  http.addHeader("Content-Type", "application/json");

  int code = http.POST(body);
  String resp = http.getString();
  http.end();

  Serial.printf("POST %s=%.3f -> %d", param, value, code);
  if (resp.length() > 0) {
    Serial.print(" | ");
    Serial.println(resp);
  } else {
    Serial.println();
  }

  return (code >= 200 && code < 300);
}

// ================= CICLO =================
void resetCycle(uint32_t now) {
  startCycle = now;
  donePH = false;
  doneEC = false;
  doneTN = false;

  Serial.println("=== NEW CYCLE ===");
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  delay(300);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  Wire.begin(I2C_SDA, I2C_SCL);

  if (!ads.begin()) {
    Serial.println("ERROR: ADS1115 no detectado");
  } else {
    Serial.println("ADS1115 OK");
    ads.setGain(GAIN_ONE);
  }

  ds.begin();
  Serial.println("DS18B20 iniciado");

  connectWiFi();
  resetCycle(millis());
}

// ================= LOOP =================
void loop() {
  uint32_t now = millis();

  // ---------- MODO CALIBRACION ----------
  if (CAL_MODE) {
    float vph = leerVoltPHProm();   // A1
    float vec = leerVoltECProm();   // A0
    float temp = leerTemp();

    Serial.printf("CAL -> V_PH=%.4f V | V_EC=%.4f V | TEMP=%.2f\n", vph, vec, temp);
    delay(2000);
    return;
  }

  // ---------- MODO NORMAL ----------
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  if (now - startCycle >= CYCLE) {
    resetCycle(now);
  }

  uint32_t elapsed = now - startCycle;

  // pH al inicio
  if (!donePH && elapsed >= T_PH) {
    float vph = leerVoltPHProm();
    float ph = convertirPH(vph);

    postMedicion("pH", ph);
    donePH = true;
  }

  // EC a los 10 min
  if (!doneEC && elapsed >= T_EC) {
    float temp = leerTemp();
    if (isnan(temp)) temp = 25.0;

    float vec = leerVoltECProm();
    float ec = convertirEC(vec, temp);

    postMedicion("EC", ec);
    doneEC = true;
  }

  // TEMP + NIVEL a los 20 min
  if (!doneTN && elapsed >= T_TN) {
    float temp = leerTemp();
    float nivel = leerNivel();

    if (!isnan(temp)) {
      postMedicion("TEMP", temp);
    } else {
      Serial.println("TEMP invalida");
    }

    if (!isnan(nivel)) {
      postMedicion("NIVEL", nivel);
    } else {
      Serial.println("NIVEL invalido");
    }

    doneTN = true;
  }

  delay(100);
}