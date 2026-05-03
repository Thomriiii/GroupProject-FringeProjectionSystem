#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ArduinoJson.h>

// ── WiFi ─────────────────────────────────────────────────────────────────────
const char* SSID     = "wompwomp";
const char* PASSWORD = "postmanpat";

// ── L298N H-bridge pins ───────────────────────────────────────────────────────
// Connect ENA and ENB on the L298N to 5V (or 3.3V) — always enabled.
// Control only via IN1–IN4.
// Raw GPIO numbers: generic ESP8266 board target does not define Dx aliases.
// Avoid strapping pins (GPIO0, GPIO2, GPIO15) to prevent boot issues.
#define PIN_IN1  4  // D2
#define PIN_IN2  14  // D5
#define PIN_IN3  12  // D6
#define PIN_IN4  13  // D7 

// ── Half-step sequence ────────────────────────────────────────────────────────
// 8 phases. Rows = phase index, columns = {IN1, IN2, IN3, IN4}
#define SEQ_LEN 8
static const uint8_t STEP_SEQ[SEQ_LEN][4] = {
    {1, 0, 0, 0},
    {1, 0, 1, 0},
    {0, 0, 1, 0},
    {0, 1, 1, 0},
    {0, 1, 0, 0},
    {0, 1, 0, 1},
    {0, 0, 0, 1},
    {1, 0, 0, 1}
};

// ── Motor / gear configuration ────────────────────────────────────────────────
// Motor: 400 half-steps/rev. Gear ratio 15:60 (1:4) → 1600 steps/plate rev.
#define STEPS_PER_REV  1600
#define DEG_PER_STEP   (360.0f / STEPS_PER_REV)  // 0.225°

// Delay between each step in ms — increase if motor skips steps
#define STEP_DELAY_MS  5

// Release coils after move to prevent L298N overheating.
// Set false if the plate must hold position under load.
#define RELEASE_AFTER_MOVE  true

// Release coils automatically after this much idle time to avoid L298N power draw.
#define IDLE_RELEASE_MS  5000

// ─────────────────────────────────────────────────────────────────────────────────
ESP8266WebServer server(80);
int   stepIndex       = 0;      // current phase in STEP_SEQ
float currentAngleDeg = 0.0f;   // dead-reckoned plate position
bool  motorIdle       = true;
unsigned long lastMotorActivityMs = 0;

// ── Angle math ────────────────────────────────────────────────────────────────
float normalizeAngle(float a) {
    a = fmod(a, 360.0f);
    return (a < 0) ? a + 360.0f : a;
}

// Signed shortest-path error: positive = CW, negative = CCW
float angularError(float current, float target) {
    float diff = target - current;
    if (diff >  180.0f) diff -= 360.0f;
    if (diff < -180.0f) diff += 360.0f;
    return diff;
}

// ── Stepper control ───────────────────────────────────────────────────────────
void setMotorActive() {
    lastMotorActivityMs = millis();
    if (motorIdle) {
        motorIdle = false;
        Serial.println("[motor] active");
    }
}

void applyPhase(int idx) {
    const uint8_t* s = STEP_SEQ[((idx % SEQ_LEN) + SEQ_LEN) % SEQ_LEN];
    setMotorActive();
    digitalWrite(PIN_IN1, s[0]);
    digitalWrite(PIN_IN2, s[1]);
    digitalWrite(PIN_IN3, s[2]);
    digitalWrite(PIN_IN4, s[3]);
}

void releaseMotor() {
    digitalWrite(PIN_IN1, LOW);
    digitalWrite(PIN_IN2, LOW);
    digitalWrite(PIN_IN3, LOW);
    digitalWrite(PIN_IN4, LOW);
    if (!motorIdle) {
        motorIdle = true;
        Serial.println("[motor] idle — coils released");
    }
    lastMotorActivityMs = millis();
}

void enterIdleIfNeeded() {
    if (!motorIdle && millis() - lastMotorActivityMs >= IDLE_RELEASE_MS) {
        releaseMotor();
    }
}

void doSteps(int count, bool cw) {
    for (int i = 0; i < count; i++) {
        stepIndex = cw ? (stepIndex + 1) % SEQ_LEN
                       : (stepIndex + SEQ_LEN - 1) % SEQ_LEN;
        applyPhase(stepIndex);
        delay(STEP_DELAY_MS);
    }
}

// Move to absolute plate angle (open-loop, dead-reckoned).
void moveTo(float targetDeg) {
    targetDeg = normalizeAngle(targetDeg);
    float error = angularError(currentAngleDeg, targetDeg);
    int steps = (int)roundf(fabsf(error) / DEG_PER_STEP);
    if (steps == 0) return;
    doSteps(steps, error > 0);
    currentAngleDeg = targetDeg;
    if (RELEASE_AFTER_MOVE) releaseMotor();
}

// ── HTTP handlers ─────────────────────────────────────────────────────────────

// POST /rotate
// Body JSON:
//   {"degrees": 90}                       → rotate 90° from current position (relative)
//   {"degrees": 270, "relative": false}   → go to absolute 270° position
void handleRotate() {
    if (!server.hasArg("plain")) {
        server.send(400, "application/json", "{\"error\":\"missing body\"}");
        return;
    }

    JsonDocument req;
    if (deserializeJson(req, server.arg("plain"))) {
        server.send(400, "application/json", "{\"error\":\"invalid JSON\"}");
        return;
    }

    if (!req.containsKey("degrees")) {
        server.send(400, "application/json", "{\"error\":\"missing field: degrees\"}");
        return;
    }

    float degrees  = req["degrees"].as<float>();
    bool  relative = req["relative"] | true;
    float target   = relative ? normalizeAngle(currentAngleDeg + degrees)
                              : normalizeAngle(degrees);

    moveTo(target);

    JsonDocument resp;
    resp["ok"]         = true;
    resp["target_deg"] = roundf(target * 100.0f) / 100.0f;
    resp["pos_deg"]    = roundf(currentAngleDeg * 100.0f) / 100.0f;

    String body;
    serializeJson(resp, body);
    server.send(200, "application/json", body);
}

// GET /position → dead-reckoned plate angle
void handlePosition() {
    JsonDocument resp;
    resp["degrees"] = roundf(currentAngleDeg * 100.0f) / 100.0f;

    String body;
    serializeJson(resp, body);
    server.send(200, "application/json", body);
}

// GET /status → health check
void handleStatus() {
    JsonDocument resp;
    resp["ip"]      = WiFi.localIP().toString();
    resp["rssi"]    = WiFi.RSSI();
    resp["pos_deg"] = roundf(currentAngleDeg * 100.0f) / 100.0f;

    String body;
    serializeJson(resp, body);
    server.send(200, "application/json", body);
}

// ── Serial command handler ────────────────────────────────────────────────────
// Commands (115200 baud, send with newline):
//   goto <deg>   — move to absolute angle
//   move <deg>   — move relative degrees (+ CW, - CCW)
//   step <n>     — step N half-steps CW (negative = CCW)
//   pos          — print dead-reckoned position
//   zero         — reset position counter to 0° without moving
//   release      — release motor coils
//   help         — print command list
static String serialBuf;

void handleSerial() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\r') continue;
        if (c == '\n') {
            serialBuf.trim();
            if (serialBuf.length() == 0) { serialBuf = ""; return; }

            String cmd = serialBuf;
            serialBuf = "";

            if (cmd.equalsIgnoreCase("pos")) {
                Serial.printf("[serial] position: %.2f°\n", currentAngleDeg);

            } else if (cmd.equalsIgnoreCase("zero")) {
                currentAngleDeg = 0.0f;
                Serial.println("[serial] position zeroed");

            } else if (cmd.equalsIgnoreCase("release")) {
                releaseMotor();
                Serial.println("[serial] motor released");

            } else if (cmd.equalsIgnoreCase("help")) {
                Serial.println("[serial] commands:");
                Serial.println("  goto <deg>  — absolute position");
                Serial.println("  move <deg>  — relative move");
                Serial.println("  step <n>    — n half-steps (neg = CCW)");
                Serial.println("  pos         — current position");
                Serial.println("  zero        — reset position to 0");
                Serial.println("  release     — release coils");

            } else if (cmd.startsWith("goto ") || cmd.startsWith("GOTO ")) {
                float target = cmd.substring(5).toFloat();
                Serial.printf("[serial] goto %.2f°\n", target);
                moveTo(target);
                Serial.printf("[serial] done — pos: %.2f°\n", currentAngleDeg);

            } else if (cmd.startsWith("move ") || cmd.startsWith("MOVE ")) {
                float delta  = cmd.substring(5).toFloat();
                float target = normalizeAngle(currentAngleDeg + delta);
                Serial.printf("[serial] move %.2f° → target %.2f°\n", delta, target);
                moveTo(target);
                Serial.printf("[serial] done — pos: %.2f°\n", currentAngleDeg);

            } else if (cmd.startsWith("step ") || cmd.startsWith("STEP ")) {
                int n = cmd.substring(5).toInt();
                if (n == 0) { Serial.println("[serial] step: bad count"); return; }
                bool cw = (n > 0);
                doSteps(abs(n), cw);
                currentAngleDeg = normalizeAngle(currentAngleDeg + n * DEG_PER_STEP);
                if (RELEASE_AFTER_MOVE) releaseMotor();
                Serial.printf("[serial] stepped %d half-steps %s — pos: %.2f°\n",
                              abs(n), cw ? "CW" : "CCW", currentAngleDeg);

            } else {
                Serial.printf("[serial] unknown command: %s\n", cmd.c_str());
            }
            return;
        }
        serialBuf += c;
    }
}

// ── Setup / Loop ──────────────────────────────────────────────────────────────
void setup() {
    delay(1000);  // Allow power to stabilize
    Serial.begin(115200);
    Serial.println("\nTurntable controller starting...");

    pinMode(PIN_IN1, OUTPUT);
    pinMode(PIN_IN2, OUTPUT);
    pinMode(PIN_IN3, OUTPUT);
    pinMode(PIN_IN4, OUTPUT);
    releaseMotor();  // Enter idle mode immediately

    WiFi.persistent(false);
    WiFi.disconnect(true);
    delay(100);
    WiFi.mode(WIFI_STA);
    WiFi.begin(SSID, PASSWORD);
    Serial.print("Connecting to WiFi");
    unsigned long wifiStart = millis();
    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - wifiStart > 15000) {
            Serial.println("\nWiFi failed — running without network");
            break;
        }
        delay(500);
        Serial.print('.');
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\nConnected — IP: %s\n", WiFi.localIP().toString().c_str());
        WiFi.setAutoReconnect(true);  // Reconnect automatically if connection drops
    }

    server.on("/rotate",   HTTP_POST, handleRotate);
    server.on("/position", HTTP_GET,  handlePosition);
    server.on("/status",   HTTP_GET,  handleStatus);
    server.begin();

    Serial.println("HTTP server ready");
    Serial.println("  POST /rotate   {\"degrees\":90}");
    Serial.println("  GET  /position");
    Serial.println("  GET  /status");
    Serial.println("Serial control: type 'help' for motor commands");
}

void loop() {
    server.handleClient();
    handleSerial();
    enterIdleIfNeeded();
}
