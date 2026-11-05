import pytest

from logic_utils import parse_guess


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_empty_input_asks_for_a_guess(raw):
    result = parse_guess(raw)
    assert result.ok is False
    assert result.error == "Enter a guess."


@pytest.mark.parametrize("raw", ["abc", "4 2", "7.9", "1e3", "--5", "0x10"])
def test_non_integer_input_is_rejected(raw):
    result = parse_guess(raw)
    assert result.ok is False
    assert result.error == "That is not a whole number."


@pytest.mark.parametrize("raw, expected", [("42", 42), (" 42 ", 42), ("+7", 7), ("-1", -1)])
def test_integers_are_accepted_with_surrounding_whitespace(raw, expected):
    result = parse_guess(raw, low=-5, high=50)
    assert result.ok is True
    assert result.value == expected
    assert result.error is None


@pytest.mark.parametrize("raw", ["0", "101"])
def test_values_outside_default_range_are_rejected(raw):
    result = parse_guess(raw)
    assert result.ok is False
    assert result.value is None
    assert "between 1 and 100" in result.error


def test_bounds_are_inclusive():
    assert parse_guess("1", low=1, high=20).ok is True
    assert parse_guess("20", low=1, high=20).ok is True


def test_range_error_uses_the_supplied_bounds():
    result = parse_guess("21", low=1, high=20)
    assert result.error == "Your guess must be between 1 and 20."
