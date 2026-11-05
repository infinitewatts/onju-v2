#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

// POD storage keeps NVS bounded; slot zero is always the home network.
struct WiFiProfile { char ssid[33] = {}; char password[65] = {}; };
struct WiFiProfiles {
    static constexpr size_t capacity = 5;
    uint32_t version = 1;
    WiFiProfile entries[capacity] = {};

    static bool credentialsValid(const char *ssid, const char *password) {
        const size_t s = strlen(ssid), p = strlen(password);
        if (!s || s > 32 || p > 64 || (p && p < 8)) return false;
        if (p == 64)
            for (size_t i = 0; i < p; ++i)
                if (!strchr("0123456789abcdefABCDEF", password[i])) return false;
        return true;
    }
    bool valid() const {
        if (version != 1 || !entries[0].ssid[0]) return false;
        for (size_t i = 0; i < capacity; ++i) {
            if (!memchr(entries[i].ssid, 0, sizeof(entries[i].ssid)) ||
                !memchr(entries[i].password, 0, sizeof(entries[i].password))) return false;
            if (!entries[i].ssid[0]) continue;
            if (!credentialsValid(entries[i].ssid, entries[i].password)) return false;
            for (size_t j = 0; j < i; ++j)
                if (!strcmp(entries[i].ssid, entries[j].ssid)) return false;
        }
        return true;
    }
    bool save(const char *ssid, const char *password, bool home = false) {
        if (!credentialsValid(ssid, password)) return false;
        WiFiProfile incoming;
        strcpy(incoming.ssid, ssid);
        strcpy(incoming.password, password);
        size_t found = capacity;
        for (size_t i = 0; i < capacity; ++i)
            if (!strcmp(entries[i].ssid, ssid)) { found = i; break; }
        if (found < capacity && !home) { entries[found] = incoming; return true; }
        const size_t slot = home ? 0 : 1;
        const size_t end = found < capacity ? found : capacity - 1;
        for (size_t i = end; i > slot; --i) entries[i] = entries[i - 1];
        entries[slot] = incoming;
        return true;
    }
    void ensureHome(const char *ssid, const char *password) {
        // Retain a home password updated by the phone; compiled credentials seed only.
        for (const auto &entry : entries)
            if (!strcmp(entry.ssid, ssid)) { save(ssid, entry.password, true); return; }
        save(ssid, password, true);
    }
};

inline uint32_t wifiAttemptBudget(uint32_t remaining, size_t attemptsLeft, bool home) {
    if (attemptsLeft <= 1) return remaining;
    const uint32_t cap = home ? 10000 : 5000;
    return remaining < cap ? remaining : cap;
}

inline bool canonicalPortalHost(const char *host, const char *ap) {
    const size_t size = strlen(ap);
    return !strncmp(host, ap, size) && (!host[size] || !strcmp(host + size, ":80"));
}

inline bool portalSubmissionAllowed(bool fromAP, const char *host, const char *origin,
                                    const char *expected, const char *submitted, const char *ap) {
    return fromAP && canonicalPortalHost(host, ap) && strlen(expected) == 32 &&
        !strcmp(expected, submitted) && (!origin[0] ||
        (!strncmp(origin, "http://", 7) && canonicalPortalHost(origin + 7, ap)));
}

inline bool homeNetworkTrusted(const char *connected, const WiFiProfiles &profiles) {
    return profiles.entries[0].ssid[0] && !strcmp(connected, profiles.entries[0].ssid);
}
