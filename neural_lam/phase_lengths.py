"""Lengths of the phases of a multi-phase training schedule.

Only the standard library is used, so the function can be tested without the
training stack.
"""


def phase_ends(lengths, defaults):
    """Epoch counts at which the phases of a schedule end.

    Parameters
    ----------
    lengths : sequence of int or None
        Epochs of each phase, as given with ``--phase_epochs``. None means
        the schedule's own lengths.
    defaults : sequence of int
        The schedule's own lengths, one per phase.

    Returns
    -------
    list of int
        Cumulative epoch counts, as Lightning counts epochs: phase ``i`` ends
        when ``ends[i]`` epochs have been trained in total. The last entry is
        the length of the whole schedule.

    A phase of 0 epochs is skipped. The schedule as a whole must have at
    least one epoch.
    """
    if lengths is None:
        lengths = defaults
    lengths = list(lengths)
    if len(lengths) != len(defaults):
        raise ValueError(
            f"this schedule has {len(defaults)} phases, but "
            f"{len(lengths)} phase lengths were given: {lengths}"
        )
    if any(int(n) != n or n < 0 for n in lengths):
        raise ValueError(
            f"phase lengths must be whole numbers of epochs, 0 or more: "
            f"{lengths}"
        )
    if sum(lengths) < 1:
        raise ValueError("the schedule needs at least one epoch")
    ends, total = [], 0
    for n in lengths:
        total += int(n)
        ends.append(total)
    return ends
