"""Device identities are exact hostname/IP pairs supplied by the operator."""
import ipaddress
import re

DEVICE_NAME = re.compile(r"onju-[0-9A-F]{6}\Z")


def validate_device_name(name: str) -> str:
    if not isinstance(name, str) or not DEVICE_NAME.fullmatch(name):
        raise ValueError("Device hostname must match onju-[0-9A-F]{6}")
    return name


def validate_device(config: dict, name: str, ip: str) -> None:
    validate_device_name(name)
    if not isinstance(ip, str):
        raise ValueError("Device IP must be a string")
    address = ipaddress.IPv4Address(ip)
    allowed = config.get("network", {}).get("allowed_devices", {})
    if str(address) != allowed.get(name):
        raise ValueError("Device hostname/IP pair is not allowlisted")
