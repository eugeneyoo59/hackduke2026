#include <Arduino.h>
#include <Adafruit_ICM20948.h>
#include <Adafruit_ICM20X.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ── BLE UUIDs ─────────────────────────────────────────────────────────────
#define SERVICE_UUID "12345678-1234-1234-1234-123456789abc"
#define CHARACTERISTIC_UUID "abcd1234-ab12-ab12-ab12-abcdef123456"

// ── FSR pins ──────────────────────────────────────────────────────────────
const int FSR_PINS[5] = {36, 39, 34, 35, 15};
float gravity[3] = {0.0f, 0.0f, 0.0f};
// ── LED ───────────────────────────────────────────────────────────────────
const int LED_PIN = 2;

// ── ICM20948 ──────────────────────────────────────────────────────────────
Adafruit_ICM20948 icm;
sensors_event_t accel_event, gyro_event, mag_event, temp_event;

// ── BLE ───────────────────────────────────────────────────────────────────
BLEServer *ble_server = nullptr;
BLECharacteristic *ble_characteristic = nullptr;
bool ble_connected = false;

class ServerCallbacks : public BLEServerCallbacks
{
  void onConnect(BLEServer *server)
  {
    ble_connected = true;
    Serial.println("BLE connected.");
  }
  void onDisconnect(BLEServer *server)
  {
    ble_connected = false;
    Serial.println("BLE disconnected.");
    delay(500);
    BLEDevice::startAdvertising();
  }
};

// ── Fall detection ────────────────────────────────────────────────────────
enum FallState
{
  FALL_IDLE,
  FALL_FREEFALL,
  FALL_IMPACT
};
FallState fall_state = FALL_IDLE;
unsigned long fall_timer = 0;

const float FREEFALL_THRESHOLD = 0.5f;
const float IMPACT_THRESHOLD = 2.5f;
const int FREEFALL_MS = 100;
const int IMPACT_WINDOW_MS = 2000;

bool detect_fall(float ax, float ay, float az)
{
  float mag = sqrt(ax * ax + ay * ay + az * az) / 9.81f;
  unsigned long now = millis();
  switch (fall_state)
  {
  case FALL_IDLE:
    if (mag < FREEFALL_THRESHOLD)
    {
      fall_state = FALL_FREEFALL;
      fall_timer = now;
    }
    break;
  case FALL_FREEFALL:
    if (mag >= FREEFALL_THRESHOLD)
    {
      if (now - fall_timer >= FREEFALL_MS)
      {
        fall_state = FALL_IMPACT;
        fall_timer = now;
      }
      else
      {
        fall_state = FALL_IDLE;
      }
    }
    if (fall_state == FALL_FREEFALL && now - fall_timer > IMPACT_WINDOW_MS)
      fall_state = FALL_IDLE;
    break;
  case FALL_IMPACT:
    if (mag > IMPACT_THRESHOLD)
    {
      fall_state = FALL_IDLE;
      return true;
    }
    if (now - fall_timer > IMPACT_WINDOW_MS)
      fall_state = FALL_IDLE;
    break;
  }
  return false;
}

// ─────────────────────────────────────────────────────────────────────────
void setup()
{
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  // BLE
  BLEDevice::init("PD-Glove");
  ble_server = BLEDevice::createServer();
  ble_server->setCallbacks(new ServerCallbacks());
  BLEService *service = ble_server->createService(SERVICE_UUID);
  ble_characteristic = service->createCharacteristic(
      CHARACTERISTIC_UUID,
      BLECharacteristic::PROPERTY_READ |
          BLECharacteristic::PROPERTY_NOTIFY);
  ble_characteristic->addDescriptor(new BLE2902());
  service->start();
  BLEDevice::getAdvertising()->addServiceUUID(SERVICE_UUID);
  BLEDevice::startAdvertising();
  Serial.println("BLE advertising as PD-Glove.");

  // ICM20948
  if (!icm.begin_I2C())
  {
    Serial.println("ICM20948 not found!");
    while (1)
      ;
  }
  icm.setAccelRange(ICM20948_ACCEL_RANGE_4_G);
  icm.setAccelRateDivisor(4095);
  Serial.println("ICM20948 ready.");

  for (int i = 0; i < 5; i++)
    pinMode(FSR_PINS[i], INPUT);
  Serial.println("Ready.");
}

// ─────────────────────────────────────────────────────────────────────────
void loop()
{
  unsigned long start = millis();

  icm.getEvent(&accel_event, &gyro_event, &temp_event, &mag_event);

  float raw_ax = accel_event.acceleration.x;
  float raw_ay = accel_event.acceleration.y;
  float raw_az = accel_event.acceleration.z;

  gravity[0] = 0.98f * gravity[0] + 0.02f * raw_ax;
  gravity[1] = 0.98f * gravity[1] + 0.02f * raw_ay;
  gravity[2] = 0.98f * gravity[2] + 0.02f * raw_az;

  float ax = raw_ax - gravity[0];
  float ay = raw_ay - gravity[1];
  float az = raw_az - gravity[2];

  bool fall = detect_fall(ax, ay, az);

  int fsr[5];
  for (int i = 0; i < 5; i++)
    fsr[i] = analogRead(FSR_PINS[i]);

  if (ble_connected)
  {
    char json[200];
    snprintf(json, sizeof(json),
             "{\"ax\":%.3f,\"ay\":%.3f,\"az\":%.3f,"
             "\"fsr\":[%d,%d,%d,%d,%d],"
             "\"fall\":%s}",
             ax, ay, az,
             fsr[0], fsr[1], fsr[2], fsr[3], fsr[4],
             fall ? "true" : "false");
    ble_characteristic->setValue((uint8_t *)json, strlen(json));
    ble_characteristic->notify();
    Serial.println(json);
  }

  if (fall)
  {
    digitalWrite(LED_PIN, HIGH);
    delay(500);
    digitalWrite(LED_PIN, LOW);
  }

  unsigned long elapsed = millis() - start;
  if (elapsed < 10)
    delay(10 - elapsed);
}