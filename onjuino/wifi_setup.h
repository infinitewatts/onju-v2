#pragma once
#include <DNSServer.h>
#include <WebServer.h>
#include <SHA2Builder.h>
#include "wifi_profiles.h"

WiFiProfiles savedNetworks;
volatile bool networkSetupActive = false;

bool storeNetworks() {
    if (!savedNetworks.valid()) return false;
    Preferences store;
    if (!store.begin("onju-wifi", false)) return false;
    WiFiProfiles readback;
    const bool unchanged = store.getBytesLength("profiles") == sizeof(readback) &&
        store.getBytes("profiles", &readback, sizeof(readback)) == sizeof(readback) &&
        !memcmp(&readback, &savedNetworks, sizeof(readback));
    const bool written = unchanged || store.putBytes("profiles", &savedNetworks, sizeof(savedNetworks)) == sizeof(savedNetworks);
    const bool verified = written && store.getBytes("profiles", &readback, sizeof(readback)) == sizeof(readback) &&
        !memcmp(&readback, &savedNetworks, sizeof(readback));
    store.end();
    if (!verified) Serial.println("ERROR: saved networks could not be verified");
    return verified;
}

void loadNetworks() {
    Preferences store;
    store.begin("onju-wifi", true);
    if (store.getBytesLength("profiles") == sizeof(savedNetworks))
        store.getBytes("profiles", &savedNetworks, sizeof(savedNetworks));
    store.end();
    if (!savedNetworks.valid()) {
        savedNetworks = WiFiProfiles{};
        savedNetworks.ensureHome(WIFI_SSID, WIFI_PASSWORD);
        savedNetworks.save(wifi_ssid.c_str(), wifi_password.c_str());
    }
    savedNetworks.ensureHome(WIFI_SSID, WIFI_PASSWORD);
    if (!storeNetworks()) Serial.println("Network profile persistence failed");
}

bool setupPasswordValid() {
    const size_t length = strlen(SETUP_PASSWORD);
    if (length < 16 || length > 63) return false;
    SHA256Builder hash;
    hash.begin(); hash.add(SETUP_PASSWORD); hash.calculate();
    return hash.toString() == OTA_PASSWORD_HASH;
}

String portalEscape(String text) {
    text.replace("&", "&amp;"); text.replace("<", "&lt;");
    text.replace(">", "&gt;"); text.replace("\"", "&quot;");
    return text;
}

bool phoneSetup() {
    if (!setupPasswordValid()) {
        Serial.println("Setup disabled: invalid setup credential");
        return false; // Never fall back to an open AP.
    }
    WiFi.disconnect(false, false);
    WiFi.mode(WIFI_AP_STA);
    String name = String(WiFi.getHostname()) + "-setup";
    if (!WiFi.softAP(name.c_str(), SETUP_PASSWORD, 1, false, 2)) {
        WiFi.mode(WIFI_STA);
        return false;
    }
    const IPAddress ap = WiFi.softAPIP();
    DNSServer dns;
    WebServer web(ap, 80); // Bind HTTP to the AP address, never the station interface.
    dns.start(53, "*", ap);
    char token[33];
    snprintf(token, sizeof(token), "%08lx%08lx%08lx%08lx", (unsigned long)esp_random(),
             (unsigned long)esp_random(), (unsigned long)esp_random(), (unsigned long)esp_random());
    bool pending = false, connected = false;
    String ssid, password, notice;
    const String canonical = ap.toString();
    const char *headers[] = {"Origin"};
    web.collectHeaders(headers, 1);
    auto fromAP = [&]() {
        return web.client().localIP() == ap &&
            ((uint32_t)web.client().remoteIP() & (uint32_t)WiFi.softAPSubnetMask()) ==
            ((uint32_t)ap & (uint32_t)WiFi.softAPSubnetMask());
    };
    auto canonicalHost = [&]() {
        return canonicalPortalHost(web.hostHeader().c_str(), canonical.c_str());
    };
    web.on("/", HTTP_GET, [&]() {
        if (!fromAP()) { web.send(403, "text/plain", "Setup AP only"); return; }
        if (!canonicalHost()) {
            web.sendHeader("Location", "http://" + canonical + "/");
            web.send(302, "text/plain", ""); return;
        }
        String page = "<!doctype html><meta name='viewport' content='width=device-width'>"
            "<title>Onju Wi-Fi setup</title><h1>Connect Onju to Wi-Fi</h1>"
            "<p>Home stays first. Up to four travel networks are remembered.</p>"
            "<form method='post' action='/save'><input name='token' type='hidden' value='";
        page += token;
        page += "'><label>Network <input name='s' list='nets' maxlength='32' required></label>"
                "<datalist id='nets'>";
        const int count = WiFi.scanNetworks();
        for (int i = 0; i < count && i < 20; ++i)
            page += "<option value=\"" + portalEscape(WiFi.SSID(i)) + "\">";
        WiFi.scanDelete();
        page += "</datalist><p><label>Password <input name='p' type='password' maxlength='64' "
                "autocomplete='new-password'></label></p><button>Save and connect</button></form>";
        page += "<p>" + portalEscape(notice) + "</p>";
        web.sendHeader("Cache-Control", "no-store");
        web.sendHeader("Content-Security-Policy", "default-src 'none'; form-action 'self'; frame-ancestors 'none'");
        web.send(200, "text/html; charset=utf-8", page);
    });
    web.on("/save", HTTP_POST, [&]() {
        const String origin = web.header("Origin");
        if (!portalSubmissionAllowed(fromAP(), web.hostHeader().c_str(), origin.c_str(),
                                     token, web.arg("token").c_str(), canonical.c_str())) {
            web.send(403, "text/plain", "Invalid setup session"); return;
        }
        if (pending) { web.send(409, "text/plain", "Connection in progress"); return; }
        ssid = web.arg("s"); password = web.arg("p");
        if (!WiFiProfiles::credentialsValid(ssid.c_str(), password.c_str())) {
            web.send(400, "text/plain", "Invalid network name or password length"); return;
        }
        pending = true;
        web.send(200, "text/plain", "Connecting. Setup closes on success; otherwise return to 192.168.4.1.");
    });
    web.onNotFound([&]() {
        if (!fromAP()) { web.send(403, "text/plain", "Setup AP only"); return; }
        web.sendHeader("Location", "http://" + ap.toString() + "/");
        web.send(302, "text/plain", "");
    });
    web.begin();
    Serial.println("Protected phone setup: " + name + " at " + ap.toString());
    const uint32_t started = millis();
    while (millis() - started < 300000 && !connected) {
        dns.processNextRequest(); web.handleClient(); delay(10);
        if (!pending) continue;
        WiFi.begin(ssid.c_str(), password.c_str());
        const uint32_t attempt = millis();
        while (WiFi.status() != WL_CONNECTED && millis() - attempt < 10000 &&
               millis() - started < 300000) { dns.processNextRequest(); delay(50); }
        connected = WiFi.status() == WL_CONNECTED && WiFi.SSID() == ssid;
        if (connected) {
            const WiFiProfiles previous = savedNetworks;
            savedNetworks.save(ssid.c_str(), password.c_str());
            if (storeNetworks()) {
                wifi_ssid = ssid; wifi_password = password;
                Serial.println("Phone network saved and verified; setup closing");
            } else {
                savedNetworks = previous;
                connected = false;
                notice = "Could not save network. Please retry.";
            }
        } else notice = "Connection failed. Check the network password and retry.";
        if (!connected) { WiFi.disconnect(false, false); pending = false; }
        password = "";
    }
    web.stop(); dns.stop(); WiFi.softAPdisconnect(true); WiFi.mode(WIFI_STA);
    Serial.println("Phone setup closed");
    return connected;
}

bool connectSavedNetworks() {
    const uint32_t started = millis();
    for (size_t i = 0; i < WiFiProfiles::capacity && millis() - started < 30000; ++i) {
        const WiFiProfile &profile = savedNetworks.entries[i];
        if (!profile.ssid[0]) continue;
        size_t left = 0;
        for (size_t j = i; j < WiFiProfiles::capacity; ++j)
            if (savedNetworks.entries[j].ssid[0]) ++left;
        const uint32_t budget = wifiAttemptBudget(30000 - (millis() - started), left, i == 0);
        WiFi.disconnect(false, false);
        WiFi.begin(profile.ssid, profile.password);
        const uint32_t attempt = millis();
        while (WiFi.status() != WL_CONNECTED && millis() - attempt < budget &&
               millis() - started < 30000) {
            if (Serial.available()) {
                const char command = Serial.read();
                if (command == 'c') enterConfigMode();
                if (command == 'r') ESP.restart();
            }
            delay(50);
        }
        if (WiFi.status() == WL_CONNECTED && WiFi.SSID() == profile.ssid) {
            wifi_ssid = profile.ssid; wifi_password = profile.password;
            return true;
        }
    }
    return false;
}

void connectDeviceWiFi() {
    networkSetupActive = true;
    WiFi.setAutoReconnect(false);
    WiFi.persistent(false);
    WiFi.mode(WIFI_STA);
    Preferences store;
    store.begin("onju-wifi", false);
    const bool forced = store.getBool("setup", false);
    store.remove("setup"); store.end();
    if (forced && phoneSetup()) { networkSetupActive = false; WiFi.setAutoReconnect(true); return; }
    while (!connectSavedNetworks()) {
        if (phoneSetup()) break;
        delay(1000);
    }
    WiFi.setAutoReconnect(true);
    networkSetupActive = false;
}
