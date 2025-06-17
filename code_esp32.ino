#include <WiFi.h> 
#include <WebServer.h>
#include <EEPROM.h>
#include <HardwareSerial.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;
float gyroZ = 0.0; 

// === MOTION COMMAND STATE MACHINE ===
enum State { IDLE, RUN_DIST, RUN_ANGLE } state = IDLE;
long targetTicks = 0;
long startEnc1 = 0, startEnc2 = 0;
float targetAngle = 0, startYaw = 0;
float yaw = 0;
unsigned long lastYawTime = 0;

// Robot parameters (tùy chỉnh theo robot của bạn)
const float WHEEL_RADIUS = 0.035;       // m
const float WHEEL_BASE   = 0.13;       // m (khoảng cách tâm hai bánh)
const int   ENCODER_PPR  = 500;        // xung/rev
const float WHEEL_CIRCUMFERENCE = 2 * PI * WHEEL_RADIUS;


// WiFi credentials mặc định
char ssid[32] = "Wifii";
char password[32] = "Nhucuong";
WiFiServer wifiServer(80);  // Server cho điều khiển động cơ
WebServer webServer(8080);  // Server cho cấu hình WiFi

// Serial communication packet specification
#define ANGLE_IDX 1
#define SPEED_LSB 2
#define SPEED_MSB 3
#define DATA_1 4
#define DATA_2 8
#define DATA_3 12
#define DATA_4 16
#define CHECKSUM_LSB 20
#define CHECKSUM_MSB 21
#define PACKET_SIZE 22
#define DATA_SIZE 7
#define BUFFER_SIZE 50

#define RX_PIN 16
#define MOTOR_PIN 5          // Dùng cho motor LIDAR (PID đã bị xóa)
#define BAUDRATE_SENSOR 115200
#define MAX_POWER 255
#define MOTOR_SPEED 20

// Định nghĩa chân cho Motor 1
#define MOTOR_IN1 18
#define MOTOR_IN2 19
#define ENA 15
#define ENCODER_PIN 4

// Định nghĩa chân cho Motor 2
#define MOTOR_IN3 23
#define MOTOR_IN4 25
#define ENB 26
#define ENCODER_PIN2 27

// Biến encoder cho bánh xe
volatile long encoderCount = 0;      // Encoder motor 1
volatile long encoderCount2 = 0;     // Encoder motor 2
// Hướng chuyển động của từng motor (1: tiến, -1: lùi, 0: dừng)
int motorDirection = 0;
int motorDirection2 = 0;

const int encoderMin = -32768;
const int encoderMax = 32767;

float gyroZBias = 0.0;
float rawGyroZ;
// Định nghĩa cho EEPROM
#define EEPROM_SIZE 128
#define SSID_ADDR 0
#define PASS_ADDR 32

// Các biến liên quan đến LIDAR
int data[DATA_SIZE];
uint8_t packet[PACKET_SIZE];
uint8_t lidarBuffer[PACKET_SIZE * BUFFER_SIZE];
int bufferIndex = 0;
const unsigned char HEAD_BYTE = 0xFA;
unsigned int packetIndex = 0;
bool waitPacket = true;
volatile bool newCommand = false;
String pendingCommand = "";

// Thêm biến PID cho góc quay
float Kp_ang = 4;
float Ki_ang = 0.3;
float Kd_ang = 0.1;
float integral_ang = 0;
float lastError_ang = 0;

const float ANGLE_TOLERANCE = 1;    
const float MAX_TURN_SPEED = 150;      // PWM max
// Thêm vào phần khai báo toàn cục
double setpointLidar = 250;  // Tốc độ mong muốn cho LIDAR
double currentSpeedLidar = 0.0;      // Tốc độ hiện tại của LIDAR
double errorLidar = 0.0;
double lastErrorLidar = 0.0;
double integralLidar = 0.0;
double derivativeLidar = 0.0;
double outputLidar = 0.0;
double Kp_lidar = 2.0;
double Ki_lidar = 0.3;
double Kd_lidar = 0.3;
unsigned long previousMillisLidar = 0;
const unsigned long lidarLOOPTIME = 50;

HardwareSerial lidarSerial(1);
WiFiClient client;

// Biến lưu tốc độ hiện tại cho motor Lidar (PWM 0-255)
int currentMotorSpeed = MOTOR_SPEED;

// ======== PHẦN THÊM: PID cho 2 động cơ bánh xe ========
// Thời gian lặp PID cho bánh xe (ms)
unsigned long previousMillisWheel = 0;
const unsigned long wheelLOOPTIME = 50;  // 50ms -> 20Hz

// Lưu giá trị encoder của lần lặp trước
volatile long encoder1Prev = 0;
volatile long encoder2Prev = 0;

// Setpoint (số xung trong vòng lặp) của từng motor
int setPointWheel1 = MOTOR_SPEED;
int setPointWheel2 = MOTOR_SPEED;

// Giá trị đo được (số xung trong vòng lặp)
int processWheel1 = 0, processWheel2 = 0;

// Các biến PID riêng cho mỗi motor bánh xe
int errorWheel1 = 0, lastErrorWheel1 = 0, dErrorWheel1 = 0;
int errorWheel2 = 0, lastErrorWheel2 = 0, dErrorWheel2 = 0;
int pidOutputWheel1 = 0, pidOutputWheel2 = 0;
// Hệ số PID (có thể điều chỉnh)
int Kp_wheel = 2;
int Kd_wheel = 0;
// (Không dùng phần tích phân cho đơn giản)

// ======== HẾT PHẦN THÊM ========

// Thêm hàm updateLidarPID
void updateLidarPID() {
    errorLidar = setpointLidar - currentSpeedLidar;
    integralLidar += errorLidar * (lidarLOOPTIME / 1000.0);
    integralLidar = constrain(integralLidar, -100, 100);
    derivativeLidar = (errorLidar - lastErrorLidar) / (lidarLOOPTIME / 1000.0);
    lastErrorLidar = errorLidar;
    outputLidar = Kp_lidar * errorLidar + Ki_lidar * integralLidar + Kd_lidar * derivativeLidar;
    outputLidar = constrain(outputLidar, 0, MAX_POWER);
    analogWrite(MOTOR_PIN, (int)outputLidar);
}

// Interrupt Service Routine cho encoder motor 1
void IRAM_ATTR encoderISR() {
    if (motorDirection == 1) {
        if (encoderCount < encoderMax) encoderCount++;
        else encoderCount = encoderMin;
    } else if (motorDirection == -1) {
        if (encoderCount > encoderMin) encoderCount--;
        else encoderCount = encoderMax;
    }
}

// Interrupt Service Routine cho encoder motor 2
void IRAM_ATTR encoderISR2() {
    if (motorDirection2 == 1) { // Tiến
        if (encoderCount2 < encoderMax) encoderCount2++;
        else encoderCount2 = encoderMin;
    } else if (motorDirection2 == -1) { // Lùi
        if (encoderCount2 > encoderMin) encoderCount2--;
        else encoderCount2 = encoderMax;
    }
}

void readEEPROM() {
    for (int i = 0; i < 32; i++) {
        ssid[i] = EEPROM.read(SSID_ADDR + i);
        password[i] = EEPROM.read(PASS_ADDR + i);
    }
}

void writeEEPROM(const char* newSSID, const char* newPass) {
    for (int i = 0; i < 32; i++) {
        EEPROM.write(SSID_ADDR + i, i < strlen(newSSID) ? newSSID[i] : 0);
        EEPROM.write(PASS_ADDR + i, i < strlen(newPass) ? newPass[i] : 0);
    }
    EEPROM.commit();
}

bool connectWiFi() {
    WiFi.begin(ssid, password);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    return WiFi.status() == WL_CONNECTED;
}

void startAPMode() {
    WiFi.mode(WIFI_AP);
    WiFi.softAP("ESP32_Config", "12345678");
    Serial.println("AP Mode started. IP: ");
    Serial.println(WiFi.softAPIP());
}

void handleRoot() {
    String page = "<!DOCTYPE html><html><body><h2>WiFi Config</h2>";
    page += "<p>Current SSID: " + String(ssid) + "</p>";
    page += "<form action=\"/save\" method=\"POST\">";
    page += "New SSID: <input type=\"text\" name=\"ssid\"><br>";
    page += "New Password: <input type=\"text\" name=\"pass\"><br>";
    page += "<input type=\"submit\" value=\"Save\"></form>";
    page += "</body></html>";
    webServer.send(200, "text/html", page);
}

void handleSave() {
    String newSSID = webServer.arg("ssid");
    String newPass = webServer.arg("pass");
    newSSID.trim();
    newPass.trim();
    writeEEPROM(newSSID.c_str(), newPass.c_str());
    strcpy(ssid, newSSID.c_str());
    strcpy(password, newPass.c_str());
    webServer.send(200, "text/html", "<h2>WiFi Saved! Rebooting...</h2>");
    delay(1000);
    ESP.restart();
}

void setup() {
    Serial.begin(115200);
    EEPROM.begin(EEPROM_SIZE);
    readEEPROM();

    WiFi.begin(ssid, password);
    pinMode(MOTOR_PIN, OUTPUT);
    lidarSerial.begin(BAUDRATE_SENSOR, SERIAL_8N1, RX_PIN, -1);
    // Điều khiển motor Lidar với tốc độ cố định
    analogWrite(MOTOR_PIN, 150);

    // Khởi tạo giao thức I2C cho MPU6050
    Wire.begin(21, 22);

    if (!mpu.begin()) {
        Serial.println("Failed to find MPU6050 chip");
        while (1) {
            delay(10);
        }
    }
    Serial.println("MPU6050 Found!");

    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
    mpu.setGyroRange(MPU6050_RANGE_250_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    Serial.println("Hiệu chỉnh gyro, vui lòng giữ hệ thống đứng yên...");
    const int calibrationSamples = 1000;
    float sumGyroZ = 0.0;
    for (int i = 0; i < calibrationSamples; i++) {
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);
        sumGyroZ += g.gyro.z;
        delay(5);
    }
    gyroZBias = sumGyroZ / calibrationSamples;
    Serial.print("Bias của gyroZ: ");
    Serial.println(gyroZBias, 4);
    Serial.println("Hiệu chỉnh hoàn tất.");

    lastYawTime = millis();

    // Cấu hình motor 1
    pinMode(MOTOR_IN1, OUTPUT);
    pinMode(MOTOR_IN2, OUTPUT);
    pinMode(ENA, OUTPUT);
    pinMode(ENCODER_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(ENCODER_PIN), encoderISR, CHANGE);

    // Cấu hình motor 2
    pinMode(MOTOR_IN3, OUTPUT);
    pinMode(MOTOR_IN4, OUTPUT);
    pinMode(ENB, OUTPUT);
    pinMode(ENCODER_PIN2, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(ENCODER_PIN2), encoderISR2, CHANGE);

    // Khi khởi động, dừng cả hai motor bánh xe
    motorStopBoth();

    if (!connectWiFi()) {
        Serial.println("\nWiFi connection failed. Starting AP mode...");
        startAPMode();
        webServer.on("/", handleRoot);
        webServer.on("/save", handleSave);
        webServer.begin();
    } else {
        Serial.println("\nWiFi connected");
        Serial.println(WiFi.localIP());
        wifiServer.begin();
        webServer.on("/", handleRoot);
        webServer.on("/save", handleSave);
        webServer.begin();
    }
    Serial.println("Encoder monitoring started");
}

// Cập nhật loop
void loop() {
    webServer.handleClient();
    handleClientCommands();
    if (!newCommand) handleLidarData();

    // Cập nhật gyroZ và tích phân thành Yaw (đơn vị: độ)
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    rawGyroZ = g.gyro.z;
    gyroZ    = rawGyroZ - gyroZBias;         // rad/s sau hiệu chỉnh bias

    // Chuyển rad/s -> °/s rồi tích phân
    const float RAD2DEG = 57.29577951308232;
    unsigned long currentTime = millis();
    float dtYaw = (currentTime - lastYawTime) / 1000.0;  // s
    float gyroDeg = gyroZ * RAD2DEG;                    // °/s
    yaw += gyroDeg * dtYaw;                             // tích phân ra °
    lastYawTime = currentTime;
    // 2) LIDAR PID
    if (currentTime - previousMillisLidar >= lidarLOOPTIME) {
        updateLidarPID();
        previousMillisLidar = currentTime;
    }

    // 3) Wheel PID
    if (currentTime - previousMillisWheel >= wheelLOOPTIME) {
        updateWheelPID();
        previousMillisWheel = currentTime;
    }

    // 4) Motion command state machine
    if (state == RUN_DIST) {
        long d1 = abs(encoderCount - startEnc1);
        long d2 = abs(encoderCount2 - startEnc2);
        long d = (d1 + d2) / 2;
        if (d >= targetTicks) {
            motorStopBoth();
            state = IDLE;
        } else {
            motorForwardBoth();
        }
    } else if (state == RUN_ANGLE) {
    float currentDelta = yaw - startYaw;
    float error = targetAngle - currentDelta;
    Serial.print("yaw: "); Serial.print(yaw);
    Serial.print(" startYaw: "); Serial.print(startYaw);
    Serial.print(" currentDelta: "); Serial.print(currentDelta);
    Serial.print(" targetAngle: "); Serial.print(targetAngle);
    Serial.print(" error: "); Serial.println(error);

    if (fabs(error) <= ANGLE_TOLERANCE) {
        motorStopBoth();
        state = IDLE;
    } else {
        float dt = (currentTime - previousMillisWheel) / 1000.0;
        previousMillisWheel = currentTime; // Cập nhật ngay sau khi dùng
        integral_ang += error * dt;
        float derivative = (error - lastError_ang) / dt;
        lastError_ang = error;
        float output = Kp_ang * error + Ki_ang * integral_ang + Kd_ang * derivative;

        // Bù dead zone nếu output quá nhỏ
if (abs(output) < 40) {
    output = (output > 0) ? 40 : -40;
}

output = constrain(output, -MAX_TURN_SPEED, MAX_TURN_SPEED);
int pwmVal = abs((int)output);

        if (output > 0) {
            // Quay sang trái: Motor1 lùi, Motor2 tiến
            digitalWrite(MOTOR_IN1, LOW);
            digitalWrite(MOTOR_IN2, HIGH);
            analogWrite(ENA, pwmVal);
            digitalWrite(MOTOR_IN3, HIGH);
            digitalWrite(MOTOR_IN4, LOW);
            analogWrite(ENB, pwmVal);
        } else {
            // Quay sang phải: Motor1 tiến, Motor2 lùi
            digitalWrite(MOTOR_IN1, HIGH);
            digitalWrite(MOTOR_IN2, LOW);
            analogWrite(ENA, pwmVal);
            digitalWrite(MOTOR_IN3, LOW);
            digitalWrite(MOTOR_IN4, HIGH);
            analogWrite(ENB, pwmVal);
        }
    }
}

    // 5) Gửi dữ liệu
    static unsigned long lastSendTime = 0;
    if (currentTime - lastSendTime >= 20) {
        if (client.connected() && bufferIndex > 0) {
            sendDataToClient();
            lastSendTime = currentTime;
            bufferIndex = 0;
        }
    }
}

void handleClientCommands() {
    if (!client || !client.connected()) {
        client = wifiServer.available();
        if (client) Serial.println("Client connected");
    }
    if (client.connected() && client.available()) {
        String cmd = client.readStringUntil('\n');
        cmd.trim();
        Serial.println("Received command: " + cmd);

        // Nếu đang thực hiện lệnh di chuyển, bỏ qua các lệnh khác
        if (state != IDLE) return;

        // Lệnh MOVE D (m)
        if (cmd.startsWith("MOVE ")) {
            float D = cmd.substring(5).toFloat();
            targetTicks = (D / WHEEL_CIRCUMFERENCE) * ENCODER_PPR;
            startEnc1 = encoderCount;
            startEnc2 = encoderCount2;
            state = RUN_DIST;

        // Lệnh ROTATE θ (deg)
        } else if (cmd.startsWith("ROTATE ")) {
        targetAngle = cmd.substring(7).toFloat()*180/PI;
        startYaw = yaw;
        // Reset các biến PID
        integral_ang = 0;
        lastError_ang = 0;
        state = RUN_ANGLE;

        // Các lệnh cũ chỉ khi IDLE
        } else if (cmd == "forward") {
            motorForwardBoth();
        } else if (cmd == "reverse") {
            motorReverseBoth();
        } else if (cmd == "left") {
            motorTurnLeft();
        } else if (cmd == "right") {
            motorTurnRight();
        } else if (cmd == "stop") {
            motorStopBoth();
        } else if (cmd == "RESET_ENCODERS") {
            encoderCount = encoderCount2 = 0;
            Serial.println("Encoders reset to 0");
        }
    }
}

void handleLidarData() {
    while (lidarSerial.available() > 0 && bufferIndex < BUFFER_SIZE * PACKET_SIZE) {
        uint8_t receivedByte = lidarSerial.read();
        if (waitPacket && receivedByte == HEAD_BYTE) {
            packetIndex = 0;
            waitPacket = false;
            packet[packetIndex++] = receivedByte;
        } else if (!waitPacket) {
            packet[packetIndex++] = receivedByte;
            if (packetIndex >= PACKET_SIZE) {
                waitPacket = true;
                decodePacket(packet, PACKET_SIZE);
                memcpy(lidarBuffer + bufferIndex, packet, PACKET_SIZE);
                bufferIndex += PACKET_SIZE;
            }
        }
    }
}

// ----- CÁC HÀM ĐIỀU KHIỂN MOTOR BÁNH XE (SỬ DỤNG PID) -----
void updateWheelPID() {
    // Tính số xung (process value) trong vòng lặp của từng motor
    processWheel1 = encoderCount - encoder1Prev;
    encoder1Prev = encoderCount;
    processWheel2 = encoderCount2 - encoder2Prev;
    encoder2Prev = encoderCount2;

    // Motor 1 PID
    errorWheel1 = setPointWheel1 - processWheel1;
    dErrorWheel1 = errorWheel1 - lastErrorWheel1;
    lastErrorWheel1 = errorWheel1;
    pidOutputWheel1 = pidOutputWheel1 + Kp_wheel * errorWheel1 + Kd_wheel * dErrorWheel1;
    pidOutputWheel1 = constrain(pidOutputWheel1, 0, MAX_POWER);

    // Motor 2 PID
    errorWheel2 = setPointWheel2 - processWheel2;
    dErrorWheel2 = errorWheel2 - lastErrorWheel2;
    lastErrorWheel2 = errorWheel2;
    pidOutputWheel2 = pidOutputWheel2 + Kp_wheel * errorWheel2 + Kd_wheel * dErrorWheel2;
    pidOutputWheel2 = constrain(pidOutputWheel2, 0, MAX_POWER);

    // Cập nhật PWM và hướng cho motor 1
    if (motorDirection == 1) {
        digitalWrite(MOTOR_IN1, HIGH);
        digitalWrite(MOTOR_IN2, LOW);
        analogWrite(ENA, pidOutputWheel1);
    } else if (motorDirection == -1) {
        digitalWrite(MOTOR_IN1, LOW);
        digitalWrite(MOTOR_IN2, HIGH);
        analogWrite(ENA, pidOutputWheel1);
    } else { // Dừng motor
        digitalWrite(MOTOR_IN1, LOW);
        digitalWrite(MOTOR_IN2, LOW);
        analogWrite(ENA, 0);
    }

    // Cập nhật PWM và hướng cho motor 2
    if (motorDirection2 == 1) {
        digitalWrite(MOTOR_IN3, HIGH);
        digitalWrite(MOTOR_IN4, LOW);
        analogWrite(ENB, pidOutputWheel2);
    } else if (motorDirection2 == -1) {
        digitalWrite(MOTOR_IN3, LOW);
        digitalWrite(MOTOR_IN4, HIGH);
        analogWrite(ENB, pidOutputWheel2);
    } else { // Dừng motor
        digitalWrite(MOTOR_IN3, LOW);
        digitalWrite(MOTOR_IN4, LOW);
        analogWrite(ENB, 0);
    }
}

// Các hàm điều khiển hướng cho 2 motor bánh xe
void motorForwardBoth() {
    // Cập nhật hướng cho motor 1 và motor 2
    motorDirection = 1;
    motorDirection2 = 1;
    // Cập nhật setpoint cho bánh xe
    setPointWheel1 = MOTOR_SPEED;
    setPointWheel2 = MOTOR_SPEED;
    Serial.println("Motors: Forward");
}

void motorReverseBoth() {
    motorDirection = -1;
    motorDirection2 = -1;
    setPointWheel1 = MOTOR_SPEED;
    setPointWheel2 = MOTOR_SPEED;
    Serial.println("Motors: Reverse");
}

void motorTurnLeft() {
    // Ví dụ: motor trái chạy chậm hơn, motor phải chạy nhanh hơn
    motorDirection = -1;
    motorDirection2 = 1;
    setPointWheel1 = MOTOR_SPEED / 2;
    setPointWheel2 = MOTOR_SPEED;
    Serial.println("Motors: Turn Left");
}

void motorTurnRight() {
    motorDirection = 1;
    motorDirection2 = -1;
    setPointWheel1 = MOTOR_SPEED;
    setPointWheel2 = MOTOR_SPEED / 2;
    Serial.println("Motors: Turn Right");
}

void motorStopBoth() {
    motorDirection = 0;
    motorDirection2 = 0;
    setPointWheel1 = 0;
    setPointWheel2 = 0;
    digitalWrite(MOTOR_IN1, LOW);
    digitalWrite(MOTOR_IN2, LOW);
    analogWrite(ENA, 0);
    digitalWrite(MOTOR_IN3, LOW);
    digitalWrite(MOTOR_IN4, LOW);
    analogWrite(ENB, 0);
    Serial.println("Motors: Stopped");
}

// ----- Các hàm xử lý dữ liệu LIDAR và checksum -----
void decodePacket(uint8_t packet[], int packetSize) {
    int data_idx = 0;
    for (int idx = 0; idx < DATA_SIZE; idx++) data[idx] = 0;
    for (int i = 0; i < packetSize; i++) {
        if (i == ANGLE_IDX) {
            int angle = (packet[i] - 0xA0) * 4;
            if (angle > 360) return;
            data[data_idx++] = angle;
        } else if (i == SPEED_LSB) {
            int speed = ((packet[SPEED_MSB] << 8) | packet[SPEED_LSB]) / 64;
            data[data_idx++] = speed;
            currentSpeedLidar = speed;  // Cập nhật tốc độ LIDAR
        } else if (i == DATA_1 || i == DATA_2 || i == DATA_3 || i == DATA_4) {
            uint16_t distance = ((packet[i+1] & 0x3F) << 8) | packet[i];
            data[data_idx++] = distance;
        }
    }
    data[data_idx] = checksum(packet, PACKET_SIZE - 2);
}

void sendDataToClient() {
    String dataString = "";
    for (int i = 0; i < DATA_SIZE; i++) {
        dataString += String(data[i]) + "\t";
    }
    dataString += String(encoderCount) + "\t" + String(encoderCount2) + "\t" + String(gyroZ, 2);
    client.println(dataString);
    Serial.println("Sent: " + dataString);
}

uint16_t checksum(uint8_t packet[], uint8_t size) {
    uint32_t chk32 = 0;
    for (int i = 0; i < size / 2; i++) {
        chk32 = (chk32 << 1) + ((packet[i * 2 + 1] << 8) + packet[i * 2]);
    }
    return (uint16_t)((chk32 & 0x7FFF) + (chk32 >> 15)) & 0x7FFF;
}