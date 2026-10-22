"""Tests for JSON extraction and text helpers."""
from src.utils.json_utils import extract_json, extract_list, parse_float, parse_score
from src.utils.text import contains_all, extract_answer, normalize_text


class TestExtractJson:
    def test_plain_object(self):
        assert extract_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}

    def test_code_fenced(self):
        text = 'Here you go:\n```json\n{"a": 1}\n```\nDone.'
        assert extract_json(text) == {"a": 1}

    def test_prose_around(self):
        text = 'The result is {"sub_problems": ["a", "b"]} as requested.'
        assert extract_json(text) == {"sub_problems": ["a", "b"]}

    def test_trailing_comma_repair(self):
        assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_nested_braces_in_strings(self):
        text = '{"note": "use {x} and }y{", "n": 3}'
        assert extract_json(text) == {"note": "use {x} and }y{", "n": 3}

    def test_no_json_returns_default(self):
        assert extract_json("no json here", default="dflt") == "dflt"
        assert extract_json(None) is None

    def test_garbage_returns_default(self):
        assert extract_json("{broken: [", default=42) == 42


class TestExtractList:
    def test_plain_list(self):
        assert extract_list('["a", "b"]') == ["a", "b"]

    def test_list_in_prose(self):
        assert extract_list('result: [1, 2, 3] ok') == [1, 2, 3]

    def test_missing_returns_default(self):
        assert extract_list("nothing", default=[]) == []


class TestParseFloat:
    def test_plain(self):
        assert parse_float("0.8") == 0.8

    def test_percent(self):
        assert parse_float("80%") == 0.8

    def test_fraction(self):
        assert parse_float("7/10") == 0.7

    def test_clamped_high(self):
        assert parse_float("1.5") == 1.0

    def test_clamped_low(self):
        assert parse_float("-0.2") == 0.0

    def test_no_number_returns_default(self):
        assert parse_float("high", default=0.25) == 0.25

    def test_none_returns_default(self):
        assert parse_float(None, default=0.5) == 0.5

    def test_custom_range(self):
        assert parse_float("7", lo=0, hi=10) == 7.0


class TestParseScore:
    def test_plain_int(self):
        assert parse_score("7") == 7

    def test_float_rounds(self):
        assert parse_score("7.6") == 8

    def test_clamped(self):
        assert parse_score("15", lo=0, hi=10) == 10
        assert parse_score("-3", lo=0, hi=10) == 0

    def test_default_on_garbage(self):
        assert parse_score("n/a", default=5) == 5


class TestExtractAnswer:
    def test_answer_marker(self):
        assert extract_answer("step 1\nstep 2\nANSWER: 42") == "42"

    def test_final_answer_marker(self):
        assert extract_answer("blah\nFINAL ANSWER: the cat") == "the cat"

    def test_last_marker_wins(self):
        text = "ANSWER: first\nlater I revised\nANSWER: second"
        assert extract_answer(text) == "second"

    def test_no_marker_last_line(self):
        assert extract_answer("line one\nline two") == "line two"

    def test_empty_lines_skipped(self):
        assert extract_answer("answer here\n\n\n") == "answer here"

    def test_none(self):
        assert extract_answer(None) == ""

    def test_marker_with_empty_rest_falls_back(self):
        # marker present but nothing after it -> falls back to last non-empty line
        assert extract_answer("stuff\nANSWER:") == "ANSWER:"


class TestNormalizeText:
    def test_lowercase_and_collapse(self):
        assert normalize_text("  Hello   WORLD \n\n") == "hello world"

    def test_none(self):
        assert normalize_text(None) == ""

    def test_contains_all(self):
        assert contains_all("The Quick Brown Fox", "quick", "fox")
        assert not contains_all("The Quick Brown Fox", "quick", "cat")
