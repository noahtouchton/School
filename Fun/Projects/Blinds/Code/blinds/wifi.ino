#include <WiFi.h>
#include <PubSubClient.h>

const char* ssid = "Physics";
const char* password = "Gators07!";

const char* mqtt_server = "192.168.4.24";
const int mqtt_port = 1883;

WiFiClient espClient;
PubSubClient client(espClient);

int Global_N = -10;

void setupWifi(){
    Serial.begin(115200);
    delay(1000);

    WiFi.begin(ssid, password);
    Serial.print("Connecting to wifi");

    while (WiFi.status() != WL_CONNECTED){
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.print("Connected. IP address: ");
    Serial.println(WiFi.localIP());

    // --- MQTT ---
    client.setServer(mqtt_server, mqtt_port);
    client.setCallback(commandCallback);
}

void reconnect() {
    while (!client.connected()) {
        Serial.print("Connecting to MQTT broker");
        if (client.connect("blindController")){
            Serial.println("Connected");
            client.subscribe("home/blinds/bedroom/set"); // Subscribe to all clients here
            client.subscribe("home/blinds/bedroom/savedN");
        } else {
            Serial.print(" failed, rc=");
            Serial.print(client.state());
            Serial.println(" retrying in 2s");
            delay(2000);
        }
    }
}

void publishN(int N){
    client.publish("home/blinds/bedroom/rotation", String(N).c_str());
    Serial.print("Published blinds on rotation: ");
    Serial.println(String(N).c_str());
}

void publishState(Bool state) {
    String pubState;
    if (state) {
        pubState = "moving";
    } else {
        pubState = "stopped";
    }
    Serial.print("Publishing: ");
    Serial.println(pubState);
    client.publish("home/blinds/bedroom/state", pubState);
}

void commandCallback(char* topic, byte* payload, unsigned int length) {
    Serial.print("Command arrived [");
    Serial.print(topic);
    Serial.print("]: ");

    String message = "";
    for (unsigned int i=0; i<length; i++) {
        message += (char)payload[i];
    }
    Serial.println(message);

    if (String(topic) == "home/blinds/bedroom/set") {
        float target = message.toFloat();
        Serial.print("Setting blind position to: ");
        Serial.println(target);
        if (Global_N != -10) {
            setPosition(target, Global_N);
        } else {
            Serial.println("No Global N stored");
        }
        
    } else if (String(topic) == "home/blinds/bedroom/savedN"){
        Global_N = target.toInt();
    }
    // Add more command handling here
}


void loopWifi(){
    if (!client.connected()){
        reconnect();
    }
    client.loop();
}