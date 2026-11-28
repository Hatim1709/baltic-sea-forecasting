"""Put a newly written checkpoint in place without ever losing the old one.

A checkpoint that is written straight to ``last.ckpt`` is unusable if the
process or the node dies while it is being written, and then the run has no
checkpoint at all. Here the new file is written under a temporary name first.
Only when it is complete does it replace ``last.ckpt``, and the file it
replaces is kept as ``last-prev.ckpt``.

At every moment at least one of ``last.ckpt`` and ``last-prev.ckpt`` is a
complete checkpoint (once the first epoch has been saved).

Only the standard library is used, so the functions can be tested without the
training stack.
"""

# Standard library
import os


def temporary_path(path):
    """Name under which the checkpoint for ``path`` is written first."""
    return f"{path}.tmp"


def previous_path(path):
    """Name under which the checkpoint replaced at ``path`` is kept."""
    root, extension = os.path.splitext(path)
    return f"{root}-prev{extension}"


def _sync(path):
    """Ask the operating system to write ``path`` to disk now.

    Some network file systems refuse this for files or folders; the rename
    is still done then.
    """
    try:
        handle = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def put_in_place(temporary, path):
    """Make the completely written file ``temporary`` the new ``path``.

    The file that was at ``path`` becomes ``previous_path(path)``. Each step
    is a rename within one folder, which replaces the target in one step.
    If the process dies between the two renames, ``path`` is missing but
    ``previous_path(path)`` and ``temporary`` are both complete.
    """
    if not os.path.isfile(temporary):
        raise FileNotFoundError(f"no new checkpoint at {temporary}")
    if os.path.getsize(temporary) == 0:
        raise OSError(f"the new checkpoint at {temporary} is empty")
    _sync(temporary)
    if os.path.exists(path):
        os.replace(path, previous_path(path))
    os.replace(temporary, path)
    _sync(os.path.dirname(os.path.abspath(path)))
