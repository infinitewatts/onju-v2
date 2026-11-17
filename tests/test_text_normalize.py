import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from pipeline.text_normalize import normalize_for_speech as n, number_to_words

NOW = datetime(2026, 11, 17, 9, 0, tzinfo=ZoneInfo("America/Chicago"))  # CST, UTC-6


class NormalizeTests(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(number_to_words(33), "thirty-three")
        self.assertEqual(number_to_words(4278), "four thousand two hundred seventy-eight")
        self.assertEqual(number_to_words(1000000), "one million")

    def test_money(self):
        self.assertEqual(n("a $4,278.60 run", NOW), "a four thousand two hundred seventy-eight dollars and sixty cents run")
        self.assertEqual(n("$20 a day", NOW), "twenty dollars a day")
        self.assertEqual(n("$1", NOW), "one dollar")

    def test_times(self):
        self.assertEqual(n("down since 20:20 UTC.", NOW), "down since two twenty P M Central time.")
        self.assertEqual(n("at 8:05 AM", NOW), "at eight oh five A M")
        self.assertEqual(n("at 4pm", NOW), "at four P M")
        self.assertEqual(n("at 12:00 PM", NOW), "at twelve P M")

    def test_ordinals_decimals_percent(self):
        self.assertEqual(n("Friday the 20th", NOW), "Friday the twentieth")
        self.assertEqual(n("the 21st and 3rd", NOW), "the twenty-first and third")
        self.assertEqual(n("1.5 kW", NOW), "one point five kilowatts")
        self.assertEqual(n("12%", NOW), "twelve percent")

    def test_abbreviations_and_ids(self):
        self.assertEqual(n("CompSource audit", NOW), "Comp Source audit")
        self.assertEqual(n("Mike called FP", NOW), "Mike called F P")
        self.assertEqual(n("ref 123456789", NOW), "ref one two three four five six seven eight nine")

    def test_plain_text_unchanged(self):
        s = "You have a few items waiting on you."
        self.assertEqual(n(s, NOW), s)


class FixTests(unittest.TestCase):
    def test_money_with_scale_and_odd_decimals(self):
        self.assertEqual(n("$1.5 million", NOW), "one point five million dollars")
        self.assertEqual(n("$12.5", NOW), "twelve dollars and fifty cents")

    def test_thousands_with_decimals(self):
        self.assertEqual(n("4,278.60", NOW), "four thousand two hundred seventy-eight point six zero")

    def test_state_and_current_transformer_are_untouched(self):
        self.assertEqual(n("Oklahoma City, OK", NOW), "Oklahoma City, OK")
        self.assertEqual(n("CT clamp", NOW), "CT clamp")

    def test_years_and_ranges(self):
        self.assertEqual(n("in 2026", NOW), "in twenty twenty-six")
        self.assertEqual(n("5-10 panels", NOW), "five to ten panels")

    def test_never_raises(self):
        huge = "$" + "9" * 4400
        self.assertEqual(n(huge, NOW), huge)


if __name__ == "__main__":
    unittest.main()
