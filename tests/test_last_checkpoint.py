"""
LastCheckpoint must hold the most recent epoch, also across several ``fit``
calls that each resume from it. That is how the multi-phase schedules in
train_model.py run: one Trainer per phase, each resuming from last.ckpt.

Runs on CPU with a tiny model. Needs pytorch-lightning.
"""

# Third-party
import pytest
import torch

pl = pytest.importorskip("pytorch_lightning")

# First-party
from neural_lam.last_checkpoint import LastCheckpoint  # noqa: E402

BATCHES = 3


class TinyModel(pl.LightningModule):
    """One linear layer; remembers which epochs it trained."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 1)
        self.epochs_trained = []

    def on_train_epoch_start(self):
        self.epochs_trained.append(self.current_epoch)

    def training_step(self, batch, batch_idx):
        return self.layer(batch[0]).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def train_dataloader(self):
        data = torch.utils.data.TensorDataset(torch.randn(BATCHES, 4))
        return torch.utils.data.DataLoader(data, batch_size=1)


def fit(model, callback, max_epochs, ckpt_path=None):
    """One phase: a new Trainer that reuses the callback, as train_model."""
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[callback],
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, ckpt_path=ckpt_path)


def saved_state(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state["epoch"], state["global_step"]


def test_last_checkpoint_follows_every_phase(tmp_path):
    torch.manual_seed(0)
    model = TinyModel()
    callback = LastCheckpoint(tmp_path / "saved")
    assert callback.last_model_path == ""

    # Phase 1: epochs 0 and 1
    fit(model, callback, max_epochs=2)
    path = callback.last_model_path
    assert path == str(tmp_path / "saved" / "last.ckpt")
    assert saved_state(path) == (1, 2 * BATCHES)

    # Phases 2 and 3 resume from last.ckpt and must continue, not restart
    fit(model, callback, max_epochs=5, ckpt_path=path)
    assert saved_state(path) == (4, 5 * BATCHES)
    fit(model, callback, max_epochs=6, ckpt_path=path)
    assert saved_state(path) == (5, 6 * BATCHES)

    assert model.epochs_trained == [0, 1, 2, 3, 4, 5]
    assert [p.name for p in (tmp_path / "saved").iterdir()] == ["last.ckpt"]


def test_checkpoint_holds_the_weights_of_its_epoch(tmp_path):
    """The file changes every epoch and matches the model at that time."""
    torch.manual_seed(0)
    model = TinyModel()
    callback = LastCheckpoint(tmp_path)
    fit(model, callback, max_epochs=1)
    first = torch.load(callback.last_model_path, weights_only=False)
    fit(model, callback, max_epochs=2, ckpt_path=callback.last_model_path)
    second = torch.load(callback.last_model_path, weights_only=False)

    assert not torch.equal(
        first["state_dict"]["layer.weight"],
        second["state_dict"]["layer.weight"],
    )
    torch.testing.assert_close(
        second["state_dict"]["layer.weight"], model.layer.weight.detach()
    )
