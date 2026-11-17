"""Spell out numbers, money, times and abbreviations before text-to-speech.

The small local TTS model garbles digits, clock times and acronyms. Everything here
turns them into plain spoken words. Times written in UTC are read as Central time.
"""
import re
from datetime import datetime
from zoneinfo import ZoneInfo

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = ((10**9, "billion"), (10**6, "million"), (10**3, "thousand"))
_ORDINAL_ENDINGS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                    "nine": "ninth", "twelve": "twelfth"}

# Spoken forms. Spelled letter by letter where people say the letters.
_ABBREVIATIONS = {
    "CompSource": "Comp Source", "UTC": "", "CT": "Central time", "CST": "Central time", "CDT": "Central time",
    "SMS": "S M S", "PR": "P R", "FP": "F P", "QBO": "Q B O", "BWS": "B W S", "API": "A P I",
    "AHJ": "A H J", "kW": "kilowatts", "kWh": "kilowatt hours", "HVAC": "H vac",
    "ID": "I D", "OK": "okay", "TTS": "T T S", "LLC": "L L C", "ASAP": "A S A P",
}


def _words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + _words(rest))
    for size, name in _SCALES:
        if n >= size:
            rest = n % size
            return _words(n // size) + " " + name + ("" if rest == 0 else " " + _words(rest))
    raise ValueError(n)


def number_to_words(n: int) -> str:
    return "minus " + _words(-n) if n < 0 else _words(n)


def _ordinal(n: int) -> str:
    base = number_to_words(n)
    head, sep, last = base.rpartition("-")
    if not sep:
        head, sep, last = base.rpartition(" ")
    word = _ORDINAL_ENDINGS.get(last) or (last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return head + sep + word


def _digits(text: str) -> str:
    return " ".join(_ONES[int(c)] for c in text)


def _money(match: re.Match) -> str:
    dollars = int(match.group(1).replace(",", ""))
    cents = match.group(2)
    out = number_to_words(dollars) + (" dollar" if dollars == 1 else " dollars")
    if cents and int(cents):
        out += " and " + number_to_words(int(cents)) + (" cent" if int(cents) == 1 else " cents")
    return out


def _clock(hour: int, minute: int) -> str:
    suffix = "A M" if hour < 12 else "P M"
    h = hour % 12 or 12
    if minute == 0:
        return f"{number_to_words(h)} {suffix}"
    spoken = ("oh " + number_to_words(minute)) if minute < 10 else number_to_words(minute)
    return f"{number_to_words(h)} {spoken} {suffix}"


def _utc_to_central(hour: int, minute: int, now: datetime) -> tuple[int, int]:
    utc = now.astimezone(ZoneInfo("UTC")).replace(hour=hour, minute=minute, second=0, microsecond=0)
    local = utc.astimezone(ZoneInfo("America/Chicago"))
    return local.hour, local.minute


def normalize_for_speech(text: str, now: datetime | None = None) -> str:
    now = now or datetime.now(ZoneInfo("America/Chicago"))

    def utc_time(m: re.Match) -> str:
        h, mi = _utc_to_central(int(m.group(1)), int(m.group(2)), now)
        return _clock(h, mi) + " Central time"

    # 20:20 UTC -> Central time
    text = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\s*UTC\b", utc_time, text)
    # 8:30 AM / 4pm / 16:45
    text = re.sub(r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*([AaPp])\.?[Mm]\.?\b",
                  lambda m: _clock(int(m.group(1)) % 12 + (12 if m.group(3) in "Pp" else 0), int(m.group(2) or 0)),
                  text)
    text = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", lambda m: _clock(int(m.group(1)), int(m.group(2))), text)
    # $4,278.60 and $20
    text = re.sub(r"\$(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?", _money, text)
    # 20th, 1st
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: _ordinal(int(m.group(1))), text)
    # 12.5 percent and 12.5%
    text = re.sub(r"\b(\d+)\.(\d+)\s*%", lambda m: f"{number_to_words(int(m.group(1)))} point {_digits(m.group(2))} percent", text)
    text = re.sub(r"\b(\d+)\s*%", lambda m: f"{number_to_words(int(m.group(1)))} percent", text)
    # decimals and thousands separators
    text = re.sub(r"\b(\d+)\.(\d+)\b", lambda m: f"{number_to_words(int(m.group(1)))} point {_digits(m.group(2))}", text)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", lambda m: number_to_words(int(m.group(0).replace(",", ""))), text)
    # long digit runs (ids, phone fragments) are read digit by digit; plain numbers as numbers
    text = re.sub(r"\b\d{7,}\b", lambda m: _digits(m.group(0)), text)
    text = re.sub(r"\b\d{1,6}\b", lambda m: number_to_words(int(m.group(0))), text)
    for short, spoken in _ABBREVIATIONS.items():
        text = re.sub(rf"(?<![A-Za-z]){re.escape(short)}(?![A-Za-z])", spoken, text)
    text = text.replace("&", " and ").replace("/", " slash ")
    return re.sub(r"\s{2,}", " ", text).strip()
