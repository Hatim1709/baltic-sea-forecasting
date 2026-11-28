"""Callback that keeps last.ckpt up to date after every training epoch."""

# Standard library
import os

# Third-party
import pytorch_lightning as pl

# Local
from .checkpoint_files import previous_path, put_in_place, temporary_path


class LastCheckpoint(pl.callbacks.Checkpoint):
    """
    Write ``<dirpath>/last.ckpt`` at the end of every training epoch.

    The multi-phase schedules in train_model.py resume each phase from
    last.ckpt, and a restarted run continues from it, so it has to hold the
    most recent epoch.

    ``ModelCheckpoint(save_top_k=0, save_last=True,
    save_on_train_epoch_end=True)`` does not do that with pytorch-lightning
    2.6.0: there the last checkpoint is only written in a step in which a
    top-k checkpoint was saved, and ``save_top_k=0`` saves none. last.ckpt was
    then written once, when the first ``fit`` ended, and every later phase
    resumed from that file.

    The checkpoint is written under a temporary name and renamed when it is
    complete, and the checkpoint of the epoch before is kept as
    ``last-prev.ckpt`` (see checkpoint_files.py). A process or node that dies
    while the file is written therefore never leaves the run without a
    usable checkpoint.

    Subclasses ``Checkpoint`` so that Lightning runs it after the other
    callbacks, like its own checkpoint callbacks.
    """

    FILE_NAME = "last.ckpt"

    def __init__(self, dirpath):
        super().__init__()
        self.dirpath = os.path.abspath(dirpath)
        # Same attribute name as ModelCheckpoint, read by train_model.py
        self.last_model_path = ""

    @property
    def previous_model_path(self):
        """Where the checkpoint of the epoch before the last one is kept."""
        return previous_path(os.path.join(self.dirpath, self.FILE_NAME))

    def on_train_epoch_end(self, trainer, pl_module):
        """Save the full training state of the epoch that just ended."""
        if trainer.fast_dev_run or trainer.sanity_checking:
            return
        if trainer.is_global_zero:
            os.makedirs(self.dirpath, exist_ok=True)
        path = os.path.join(self.dirpath, self.FILE_NAME)
        temporary = temporary_path(path)
        # Writes on rank zero only, and returns on every rank only after the
        # file has been written
        trainer.save_checkpoint(temporary)
        if trainer.is_global_zero:
            put_in_place(temporary, path)
        self.last_model_path = path
