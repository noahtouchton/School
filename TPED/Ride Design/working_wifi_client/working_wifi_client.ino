#include <WiFiS3.h>

/////////////////////// EDIT THESE ///////////////////////
const char* SSID = "TPED";
const char* PASS = "TPEDwifi";
//////////////////////////////////////////////////////////

WiFiServer server(80);

static unsigned long lastStatusPrint = 0;
static unsigned long lastRetry = 0;

String ipToString(IPAddress ip) {
  return String(ip[0]) + "." + String(ip[1]) + "." + String(ip[2]) + "." + String(ip[3]);
}

String macToString() {
  byte mac[6];
  WiFi.macAddress(mac);
  String s = "";
  for (int i = 0; i < 6; i++) {
    if (i) s += ":";
    if (mac[i] < 16) s += "0";
    s += String(mac[i], HEX);
  }
  s.toUpperCase();
  return s;
}

void printNetInfo() {
  Serial.println("\n--- Network Info ---");
  Serial.print("WiFi.status(): "); Serial.println(WiFi.status());
  Serial.print("SSID: "); Serial.println(WiFi.SSID());
  Serial.print("RSSI: "); Serial.print(WiFi.RSSI()); Serial.println(" dBm");
  Serial.print("MAC: "); Serial.println(macToString());

  Serial.print("IP: "); Serial.println(ipToString(WiFi.localIP()));
  Serial.print("Gateway: "); Serial.println(ipToString(WiFi.gatewayIP()));
  Serial.print("Subnet: "); Serial.println(ipToString(WiFi.subnetMask()));
  Serial.println("--------------------\n");
}

bool connectAndGetDhcp(unsigned long connectTimeoutMs = 20000, unsigned long dhcpTimeoutMs = 15000) {
  Serial.print("Connecting to SSID: ");
  Serial.println(SSID);

  WiFi.disconnect();
  delay(200);

  WiFi.begin(SSID, PASS);

  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < connectTimeoutMs) {
    Serial.print(".");
    delay(500);
  }
  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("FAILED: did not reach WL_CONNECTED.");
    Serial.print("WiFi.status() = "); Serial.println(WiFi.status());
    return false;
  }

  Serial.println("Associated (WL_CONNECTED). Waiting for DHCP IP...");

  unsigned long t1 = millis();
  while (WiFi.localIP() == IPAddress(0, 0, 0, 0) && millis() - t1 < dhcpTimeoutMs) {
    Serial.print(".");
    delay(500);
  }
  Serial.println();

  if (WiFi.localIP() == IPAddress(0, 0, 0, 0)) {
    Serial.println("FAILED: DHCP did not assign an IP (still 0.0.0.0).");
    printNetInfo();
    return false;
  }

  Serial.println("SUCCESS: got DHCP IP!");
  printNetInfo();
  return true;
}

void sendHttpHeader(WiFiClient& c, const char* contentType) {
  c.println("HTTP/1.1 200 OK");
  c.print("Content-Type: ");
  c.println(contentType);
  c.println("Connection: close");
  c.println();
}

String readRequestLine(WiFiClient& client) {
  String line = "";
  unsigned long start = millis();
  while (client.connected() && millis() - start < 2000) {
    while (client.available()) {
      char ch = client.read();
      if (ch == '\r') continue;
      if (ch == '\n') return line;
      line += ch;
    }
  }
  return line;
}

String parsePathFromGetLine(const String& reqLine) {
  // Example: "GET /api HTTP/1.1"
  if (!reqLine.startsWith("GET ")) return "/";
  int s = 4;
  int e = reqLine.indexOf(' ', s);
  if (e <= s) return "/";
  return reqLine.substring(s, e);
}

void handleRoot(WiFiClient& c) {
  sendHttpHeader(c, "text/html; charset=utf-8");

  int led = digitalRead(LED_BUILTIN);

  c.println("<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'>");
  c.println("<title>UNO R4 WiFi</title></head><body style='font-family:system-ui;padding:16px;'>");
  c.println("<h2>UNO R4 WiFi Server</h2>");

  c.print("<p><b>Status:</b> ");
  c.print(WiFi.status() == WL_CONNECTED ? "WL_CONNECTED" : "NOT CONNECTED");
  c.println("</p>");

  c.print("<p><b>SSID:</b> "); c.print(WiFi.SSID()); c.println("</p>");
  c.print("<p><b>IP:</b> "); c.print(ipToString(WiFi.localIP())); c.println("</p>");
  c.print("<p><b>Gateway:</b> "); c.print(ipToString(WiFi.gatewayIP())); c.println("</p>");
  c.print("<p><b>Subnet:</b> "); c.print(ipToString(WiFi.subnetMask())); c.println("</p>");
  c.print("<p><b>RSSI:</b> "); c.print(WiFi.RSSI()); c.println(" dBm</p>");
  c.print("<p><b>MAC:</b> "); c.print(macToString()); c.println("</p>");

  c.print("<p><b>LED:</b> "); c.print(led ? "ON" : "OFF"); c.println("</p>");
  c.println("<p>");
  c.println("<a href='/led/on'><button style='padding:10px 14px;'>LED ON</button></a> ");
  c.println("<a href='/led/off'><button style='padding:10px 14px;'>LED OFF</button></a>");
  c.println("</p>");

  c.print("<p><b>A0:</b> "); c.print(analogRead(A0)); c.println("</p>");

  c.println("<hr>");
  c.println("<p>Endpoints:</p>");
  c.println("<ul>");
  c.println("<li><code>/</code> (this page)</li>");
  c.println("<li><code>/api</code> (JSON status)</li>");
  c.println("<li><code>/led/on</code>, <code>/led/off</code></li>");
  c.println("<li><code>/reconnect</code> (force reconnect + DHCP)</li>");
  c.println("</ul>");

  c.println("</body></html>");
}

void handleApi(WiFiClient& c) {
  sendHttpHeader(c, "application/json; charset=utf-8");

  int led = digitalRead(LED_BUILTIN);

  c.print("{\"ssid\":\"");
  c.print(WiFi.SSID());
  c.print("\",\"status\":");
  c.print(WiFi.status());
  c.print(",\"ip\":\"");
  c.print(ipToString(WiFi.localIP()));
  c.print("\",\"gateway\":\"");
  c.print(ipToString(WiFi.gatewayIP()));
  c.print("\",\"subnet\":\"");
  c.print(ipToString(WiFi.subnetMask()));
  c.print("\",\"rssi\":");
  c.print(WiFi.RSSI());
  c.print(",\"mac\":\"");
  c.print(macToString());
  c.print("\",\"led\":");
  c.print(led ? "true" : "false");
  c.print(",\"a0\":");
  c.print(analogRead(A0));
  c.println("}");
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  Serial.begin(115200);
  delay(1500);

  Serial.println("\n==============================");
  Serial.println("UNO R4 WiFi HTTP Server Test");
  Serial.println("==============================");

  Serial.print("WiFi firmware: ");
  Serial.println(WiFi.firmwareVersion());

  bool ok = connectAndGetDhcp();
  if (!ok) {
    Serial.println("Initial connect failed. Will keep retrying in loop.");
  }

  server.begin();
  Serial.println("HTTP server started on port 80");
  Serial.println("Open the printed IP in your browser.");
}

void loop() {
  // Print status every 5 seconds
  if (millis() - lastStatusPrint > 5000) {
    lastStatusPrint = millis();
    Serial.print("loop | status=");
    Serial.print(WiFi.status());
    Serial.print(" | ip=");
    Serial.println(ipToString(WiFi.localIP()));
  }

  // Retry connect/DHCP every 10 seconds if needed
  if ((WiFi.status() != WL_CONNECTED || WiFi.localIP() == IPAddress(0,0,0,0)) &&
      millis() - lastRetry > 10000) {
    lastRetry = millis();
    Serial.println("Retrying Wi-Fi/DHCP...");
    connectAndGetDhcp();
  }

  // HTTP handling
  WiFiClient client = server.available();
  if (!client) return;

  String reqLine = readRequestLine(client);
  String path = parsePathFromGetLine(reqLine);

  Serial.print("HTTP request: ");
  Serial.println(reqLine);

  if (path == "/led/on") {
    digitalWrite(LED_BUILTIN, HIGH);
    handleRoot(client);
  } else if (path == "/led/off") {
    digitalWrite(LED_BUILTIN, LOW);
    handleRoot(client);
  } else if (path == "/api") {
    handleApi(client);
  } else if (path == "/reconnect") {
    sendHttpHeader(client, "text/plain; charset=utf-8");
    client.println("Reconnecting now. Check Serial for details.");
    client.stop();
    delay(50);
    connectAndGetDhcp();
    return;
  } else {
    handleRoot(client);
  }

  delay(5);
  client.stop();
}