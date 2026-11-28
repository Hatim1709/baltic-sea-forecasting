"""
Putting a new checkpoint in place must never leave a run without a complete
checkpoint, whatever moment the process dies at.

Needs only the standard library.
"""

# Third-party
import pytest

# First-party
from neural_lam.checkpoint_files import (
    previous_path,
    put_in_place,
    temporary_path,
)


def write(path, text):
    with open(path, "w", encoding="utf-8") as out:
        out.write(text)


def read(path):
    with open(path, encoding="utf-8") as source:
        return source.read()


def test_names(tmp_path):
    path = str(tmp_path / "last.ckpt")
    assert temporary_path(path) == str(tmp_path / "last.ckpt.tmp")
    assert previous_path(path) == str(tmp_path / "last-prev.ckpt")


def test_first_checkpoint(tmp_path):
    path = str(tmp_path / "last.ckpt")
    write(temporary_path(path), "epoch 0")
    put_in_place(temporary_path(path), path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["last.ckpt"]
    assert read(path) == "epoch 0"


def test_later_checkpoints_keep_exactly_one_previous(tmp_path):
    path = str(tmp_path / "last.ckpt")
    for epoch in range(4):
        write(temporary_path(path), f"epoch {epoch}")
        put_in_place(temporary_path(path), path)
        assert read(path) == f"epoch {epoch}"
        if epoch:
            assert read(previous_path(path)) == f"epoch {epoch - 1}"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "last-prev.ckpt",
        "last.ckpt",
    ]


def test_death_while_writing_keeps_the_old_checkpoint(tmp_path):
    """A half-written temporary file does not touch last.ckpt, and the next
    complete checkpoint simply overwrites the leftover."""
    path = str(tmp_path / "last.ckpt")
    write(temporary_path(path), "epoch 0")
    put_in_place(temporary_path(path), path)

    write(temporary_path(path), "epo")  # the process died here
    assert read(path) == "epoch 0"

    write(temporary_path(path), "epoch 1")
    put_in_place(temporary_path(path), path)
    assert read(path) == "epoch 1"
    assert read(previous_path(path)) == "epoch 0"


def test_missing_or_empty_new_checkpoint_changes_nothing(tmp_path):
    path = str(tmp_path / "last.ckpt")
    write(path, "epoch 0")
    with pytest.raises(FileNotFoundError):
        put_in_place(temporary_path(path), path)
    write(temporary_path(path), "")
    with pytest.raises(OSError):
        put_in_place(temporary_path(path), path)
    assert read(path) == "epoch 0"
    assert not (tmp_path / "last-prev.ckpt").exists()
