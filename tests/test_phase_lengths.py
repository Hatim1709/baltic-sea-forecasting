"""
--phase_epochs must leave the schedules as they were when it is not given,
and reject lengths that do not fit the schedule.

Needs only the standard library.
"""

# Third-party
import pytest

# First-party
from neural_lam.phase_lengths import phase_ends

PRETRAIN = (100, 200, 25)
PROBABILISTIC = (100, 200, 25, 25)


def test_defaults_are_the_schedules_as_before():
    assert phase_ends(None, PRETRAIN) == [100, 300, 325]
    assert phase_ends(None, PROBABILISTIC) == [100, 300, 325, 350]


def test_given_lengths():
    assert phase_ends([6, 8, 1], PRETRAIN) == [6, 14, 15]
    assert phase_ends((2, 2, 1, 1), PROBABILISTIC) == [2, 4, 5, 6]


def test_a_phase_of_zero_epochs_is_allowed():
    assert phase_ends([0, 5, 0], PRETRAIN) == [0, 5, 5]


@pytest.mark.parametrize(
    "lengths",
    [[6, 8], [6, 8, 1, 1], [], [6, -1, 1], [0, 0, 0], [1.5, 2, 1]],
)
def test_rejected(lengths):
    with pytest.raises(ValueError):
        phase_ends(lengths, PRETRAIN)
