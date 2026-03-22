#include <Arduino.h>
#include <Adafruit_ICM20948.h>
#include <Adafruit_ICM20X.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

// ── BLE UUIDs ─────────────────────────────────────────────────────────────
#define SERVICE_UUID "12345678-1234-1234-1234-123456789abc"
#define CHARACTERISTIC_UUID "abcd1234-ab12-ab12-ab12-abcdef123456"

// ── Normalization constants ───────────────────────────────────────────────
const float MEAN[3] = {0.000294f, 0.001564f, 0.001676f};
const float STD[3] = {0.028253f, 0.055107f, 0.046798f};
const float THRESHOLD = 0.4f;

// ── Voting ────────────────────────────────────────────────────────────────
const int VOTE_WINDOW = 10;
const int ALERT_COUNT = 6;

// ── FSR pins ──────────────────────────────────────────────────────────────
const int FSR_PINS[5] = {36, 39, 34, 35, 15};

// ── Model settings ────────────────────────────────────────────────────────
const int WINDOW_SIZE = 200;
const int N_CHANNELS = 3;

// ── TFLite ────────────────────────────────────────────────────────────────
constexpr int kTensorArenaSize = 45 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// ── ICM20948 ──────────────────────────────────────────────────────────────
Adafruit_ICM20948 icm;
sensors_event_t accel_event, gyro_event, mag_event, temp_event;

// ── Ring buffer ───────────────────────────────────────────────────────────
float ring_buffer[WINDOW_SIZE][N_CHANNELS];
int buffer_index = 0;
bool buffer_filled = false;

// ── Vote buffer ───────────────────────────────────────────────────────────
int vote_buffer[VOTE_WINDOW] = {0};
int vote_index = 0;

// ── LED ───────────────────────────────────────────────────────────────────
const int LED_PIN = 2;

// ── BLE ───────────────────────────────────────────────────────────────────
BLEServer *ble_server = nullptr;
BLECharacteristic *ble_characteristic = nullptr;
bool ble_connected = false;

class ServerCallbacks : public BLEServerCallbacks
{
  void onConnect(BLEServer *server)
  {
    ble_connected = true;
  }
  void onDisconnect(BLEServer *server)
  {
    ble_connected = false;
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
    if (fall_state == FALL_FREEFALL &&
        now - fall_timer > IMPACT_WINDOW_MS)
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
  Serial.println("BLE ready.");

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

  // TFLite
  const tflite::Model *model = tflite::GetModel(pd_model);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    Serial.println("Schema mismatch!");
    while (1)
      ;
  }

  static tflite::MicroErrorReporter micro_error_reporter;
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize,
      &micro_error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    Serial.println("AllocateTensors failed!");
    while (1)
      ;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("Model loaded. Ready.");
}

// ─────────────────────────────────────────────────────────────────────────
float normalize(float value, int channel)
{
  return (value - MEAN[channel]) / STD[channel];
}

// ─────────────────────────────────────────────────────────────────────────
float run_inference()
{
  float *input_data = input->data.f;

  for (int i = 0; i < WINDOW_SIZE; i++)
  {
    int src = (buffer_index + i) % WINDOW_SIZE;
    for (int c = 0; c < N_CHANNELS; c++)
    {
      input_data[i * N_CHANNELS + c] = normalize(ring_buffer[src][c], c);
    }
  }

  if (interpreter->Invoke() != kTfLiteOk)
    return 0.0f;

  return interpreter->output(0)->data.f[0];
}

// ─────────────────────────────────────────────────────────────────────────
bool majority_vote(int is_pd)
{
  vote_buffer[vote_index] = is_pd;
  vote_index = (vote_index + 1) % VOTE_WINDOW;
  int count = 0;
  for (int i = 0; i < VOTE_WINDOW; i++)
    count += vote_buffer[i];
  return count >= ALERT_COUNT;
}

// ─────────────────────────────────────────────────────────────────────────
void loop()
{
  unsigned long start = millis();

  icm.getEvent(&accel_event, &gyro_event, &temp_event, &mag_event);

  float ax = accel_event.acceleration.x;
  float ay = accel_event.acceleration.y;
  float az = accel_event.acceleration.z;

  bool fall_detected = detect_fall(ax, ay, az);

  ring_buffer[buffer_index][0] = ax;
  ring_buffer[buffer_index][1] = ay;
  ring_buffer[buffer_index][2] = az;
  buffer_index++;

  if (buffer_index >= WINDOW_SIZE)
  {
    buffer_index = 0;
    buffer_filled = true;
  }

  int fsr[5];
  for (int i = 0; i < 5; i++)
    fsr[i] = analogRead(FSR_PINS[i]);

  float risk_score = 0.0f;
  bool pd_alert = false;

  if (buffer_filled && buffer_index == 0)
  {
    float probability = run_inference();
    risk_score = probability * 100.0f;
    pd_alert = majority_vote((probability > THRESHOLD) ? 1 : 0);
    digitalWrite(LED_PIN, pd_alert ? HIGH : LOW);
  }

  if (ble_connected)
  {
    char json[200];
    snprintf(json, sizeof(json),
             "{\"ax\":%.2f,\"ay\":%.2f,\"az\":%.2f,"
             "\"fsr\":[%d,%d,%d,%d,%d],"
             "\"risk\":%.1f,\"pd\":%s,\"fall\":%s}",
             ax, ay, az,
             fsr[0], fsr[1], fsr[2], fsr[3], fsr[4],
             risk_score,
             pd_alert ? "true" : "false",
             fall_detected ? "true" : "false");
    ble_characteristic->setValue((uint8_t *)json, strlen(json));
    ble_characteristic->notify();
  }

  unsigned long elapsed = millis() - start;
  if (elapsed < 10)
    delay(10 - elapsed);
}