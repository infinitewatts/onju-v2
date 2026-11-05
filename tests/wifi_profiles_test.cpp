#include "../onjuino/wifi_profiles.h"
#include <cassert>
#include <cstdio>

int main() {
    WiFiProfiles profiles;
    assert(!profiles.valid());
    profiles.ensureHome("Home", "home-password");
    assert(profiles.valid());
    assert(profiles.save("Travel1", "travel-password"));
    assert(profiles.save("Travel2", "travel-password"));
    assert(profiles.save("Travel3", "travel-password"));
    assert(profiles.save("Travel4", "travel-password"));
    assert(profiles.save("Travel5", "travel-password"));
    assert(!strcmp(profiles.entries[0].ssid, "Home"));
    assert(!strcmp(profiles.entries[1].ssid, "Travel5"));
    assert(!strcmp(profiles.entries[4].ssid, "Travel2"));
    assert(profiles.valid());
    assert(profiles.save("Home", "new-home-password"));
    profiles.ensureHome("Home", "stale-compiled-password");
    assert(!strcmp(profiles.entries[0].password, "new-home-password"));
    WiFiProfiles recovered;
    memcpy(&recovered, &profiles, sizeof(profiles));
    assert(recovered.valid());
    assert(!strcmp(recovered.entries[1].ssid, "Travel5"));
    assert(homeNetworkTrusted("Home", profiles));
    assert(!homeNetworkTrusted("Travel5", profiles));
    assert(!homeNetworkTrusted("OtherLANWithSameIP", profiles));
    WiFiProfiles before = profiles;
    assert(!profiles.save("", "password"));
    assert(!profiles.save("Network", "short"));
    assert(!profiles.save("123456789012345678901234567890123", "password"));
    assert(!memcmp(&before, &profiles, sizeof(profiles)));
    assert(WiFiProfiles::credentialsValid("OpenNetwork", ""));
    assert(WiFiProfiles::credentialsValid("Network", "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"));
    assert(!WiFiProfiles::credentialsValid("Network", "z123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"));
    recovered.version = 2;
    assert(!recovered.valid());
    recovered = profiles;
    memset(recovered.entries[1].ssid, 'X', sizeof(recovered.entries[1].ssid));
    assert(!recovered.valid());
    recovered = profiles;
    recovered.entries[1] = recovered.entries[0];
    assert(!recovered.valid());
    profiles.ensureHome("Travel3", "new-default");
    assert(!strcmp(profiles.entries[0].ssid, "Travel3"));
    assert(profiles.valid());
    uint32_t remaining = 30000;
    for (size_t i = 0; i < 5; ++i) remaining -= wifiAttemptBudget(remaining, 5 - i, i == 0);
    assert(remaining == 0);
    assert(wifiAttemptBudget(30000, 1, true) == 30000);
    assert(wifiAttemptBudget(2000, 3, false) == 2000);
    const char *ap = "192.168.4.1", *token = "0123456789abcdef0123456789abcdef";
    assert(canonicalPortalHost(ap, ap));
    assert(canonicalPortalHost("192.168.4.1:80", ap));
    assert(!canonicalPortalHost("192.168.4.1.evil.test", ap));
    assert(!canonicalPortalHost("evil.test", ap));
    assert(!canonicalPortalHost("192.168.4.1:81", ap));
    assert(portalSubmissionAllowed(true, ap, "http://192.168.4.1", token, token, ap));
    assert(portalSubmissionAllowed(true, ap, "", token, token, ap));
    assert(!portalSubmissionAllowed(false, ap, "", token, token, ap));
    assert(!portalSubmissionAllowed(true, "evil.test", "http://evil.test", token, token, ap));
    assert(!portalSubmissionAllowed(true, ap, "http://evil.test", token, token, ap));
    assert(!portalSubmissionAllowed(true, ap, "null", token, token, ap));
    assert(!portalSubmissionAllowed(true, ap, "", token, "wrong", ap));
    puts("PASS: profile persistence/order/bounds, 30s budget, travel pin and portal rebinding/CSRF policy");
}
