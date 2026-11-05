"""Keep the OTA password in macOS Keychain, never argv/stdout or the repository.

The setup AP needs the password plus the OTA SHA-256 verifier in an ignored,
owner-only firmware header. Neither is printed.
Uses Security.framework directly so no secret crosses a subprocess boundary.
"""
import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
from datetime import date

SERVICE = "Onju Voice OTA"


def password(device: str, create: bool = False) -> bytes:
    if not re.fullmatch(r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", device):
        raise ValueError("Expected a device MAC address")
    cf = c.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    sec = c.CDLL("/System/Library/Frameworks/Security.framework/Security")
    ptr = c.c_void_p
    cf.CFStringCreateWithCString.argtypes = [ptr, c.c_char_p, c.c_uint32]
    cf.CFStringCreateWithCString.restype = ptr
    cf.CFDictionaryCreate.argtypes = [ptr, c.POINTER(ptr), c.POINTER(ptr), c.c_long, ptr, ptr]
    cf.CFDictionaryCreate.restype = ptr
    cf.CFDataCreate.argtypes = [ptr, ptr, c.c_long]
    cf.CFDataCreate.restype = ptr
    cf.CFDataGetLength.argtypes = [ptr]
    cf.CFDataGetLength.restype = c.c_long
    cf.CFDataGetBytePtr.argtypes = [ptr]
    cf.CFDataGetBytePtr.restype = ptr
    sec.SecItemCopyMatching.argtypes = [ptr, c.POINTER(ptr)]
    sec.SecItemCopyMatching.restype = c.c_int32
    sec.SecItemAdd.argtypes = [ptr, ptr]
    sec.SecItemAdd.restype = c.c_int32

    def symbol(lib, name):
        return ptr.in_dll(lib, name).value

    def string(value):
        return cf.CFStringCreateWithCString(None, value.encode(), 0x08000100)

    def dictionary(items):
        return cf.CFDictionaryCreate(None, (ptr * len(items))(*items),
                                     (ptr * len(items))(*items.values()), len(items), None, None)

    base = {symbol(sec, "kSecClass"): symbol(sec, "kSecClassGenericPassword"),
            symbol(sec, "kSecAttrService"): string(SERVICE),
            symbol(sec, "kSecAttrAccount"): string(device.upper())}
    query = dict(base)
    query[symbol(sec, "kSecReturnData")] = symbol(cf, "kCFBooleanTrue")
    query[symbol(sec, "kSecMatchLimit")] = symbol(sec, "kSecMatchLimitOne")
    data = ptr()
    result = sec.SecItemCopyMatching(dictionary(query), c.byref(data))
    if result == 0:
        return c.string_at(cf.CFDataGetBytePtr(data), cf.CFDataGetLength(data))
    if result != -25300 or not create:
        raise RuntimeError(f"OTA Keychain read failed (status {result}); no value returned")
    value = secrets.token_urlsafe(32).encode()
    base[symbol(sec, "kSecValueData")] = cf.CFDataCreate(None, c.c_char_p(value), len(value))
    result = sec.SecItemAdd(dictionary(base), None)
    if result != 0:
        raise RuntimeError(f"OTA Keychain creation failed (status {result})")
    # Verify authoritative readback before using the generated credential.
    if password(device) != value:
        raise RuntimeError("OTA Keychain readback mismatch")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--setup-qr", action="store_true", help="Write an owner-only Wi-Fi QR SVG; never print its data")
    args = parser.parse_args()
    value = password(args.device, create=True)
    if not re.fullmatch(rb"[A-Za-z0-9_-]{43}", value):
        raise RuntimeError("Stored OTA/setup credential has an unexpected format; no rotation performed")
    target = Path(__file__).resolve().parents[1] / "onjuino/ota_credentials.h"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write('#pragma once\n#define OTA_PASSWORD_HASH "' + hashlib.sha256(value).hexdigest() + '"\n')
        f.write('#define SETUP_PASSWORD ' + json.dumps(value.decode("ascii")) + '\n')
    print("OTA/setup credential stored/reused in Keychain; private firmware header prepared.")
    if args.setup_qr:
        import qrcode
        import qrcode.image.svg
        suffix = args.device.replace(":", "")[-6:].upper()
        # Generated passwords are URL-safe, so contain no QR field separators.
        data = "WIFI:T:WPA;S:onju-" + suffix + "-setup;P:" + value.decode("ascii") + ";H:false;;"
        image = qrcode.make(data, image_factory=qrcode.image.svg.SvgPathImage)
        folder = target.parents[1] / "data"
        folder.mkdir(exist_ok=True)
        output = folder / (date.today().isoformat() + "__onju-" + suffix.lower() + "-setup__qr.svg")
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            image.save(f)
        print("Private setup QR saved:", output)


if __name__ == "__main__":
    main()
