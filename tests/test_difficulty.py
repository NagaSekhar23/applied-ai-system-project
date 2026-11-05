import pytest

from logic_utils import (
    DIFFICULTY_CONFIGS,
    get_attempt_limit,
    get_difficulty_config,
    get_range_for_difficulty,
)


@pytest.mark.parametrize(
    "difficulty, expected_range, expected_attempts",
    [("Easy", (1, 20), 6), ("Normal", (1, 100), 8), ("Hard", (1, 50), 5)],
)
def test_each_difficulty_has_its_range_and_attempt_limit(
    difficulty, expected_range, expected_attempts
):
    assert get_range_for_difficulty(difficulty) == expected_range
    assert get_attempt_limit(difficulty) == expected_attempts


def test_config_table_is_the_single_source_of_truth():
    config = get_difficulty_config("Hard")
    assert (config.low, config.high, config.max_attempts) == (1, 50, 5)
    assert set(DIFFICULTY_CONFIGS) == {"Easy", "Normal", "Hard"}


@pytest.mark.parametrize("bad", ["", "normal", "Insane", None])
def test_unknown_difficulty_fails_loudly(bad):
    with pytest.raises(ValueError, match="Unknown difficulty"):
        get_difficulty_config(bad)
