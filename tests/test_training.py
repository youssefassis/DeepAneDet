import pytest

from deepanedet.training.dataset import getDataloaders
from deepanedet.training.losses import get_loss_criterion
from deepanedet.training.trainer import create_trainer
from deepanedet.utils import get_model, get_optimizer_scheduler


@pytest.fixture
def model_config(training_config, tmp_path):
    """Config of a tiny model trained on CPU, with its checkpoints in tmp_path."""
    return {
        **training_config,
        "model": "ASSIS",
        "in_channels": 1,
        "n_base_filters": 4,
        "layer_order": "cbl",
        "depth": 4,
        "basic_module": "ExtResNetBlock",
        "optimizer": "SGD",
        "momentum": 0.9,
        "weight_decay": 0,
        "lr": "WarmupPoly",
        "warmEpochs": 1,
        "initial_learning_rate": 1e-2,
        "min_lr": 1e-4,
        "start_epoch": 1,
        "stop_epoch": 4,
        "save model each n epoch": 10,
        "model_file": str(tmp_path / "last_checkpoint.pytorch"),
    }


@pytest.fixture
def train(patient_db, model_config):
    """Trains (or resumes training) the tiny model up to n_epochs and returns its trainer; overrides update the config."""

    def run(n_epochs, **overrides):
        config = {**model_config, "n_epochs": n_epochs, **overrides}
        loaders, iterations = getDataloaders(patient_db, None, config, workers=0)
        model = get_model(config)
        optimizer, scheduler = get_optimizer_scheduler(
            model, config, warmup_steps=iterations["train"] * config["warmEpochs"]
        )
        trainer = create_trainer(
            config,
            "cpu",
            model,
            optimizer,
            scheduler,
            get_loss_criterion("YOLO_Loss"),
            loaders,
            iterations,
            config["scales"],
            None,
        )
        trainer.fit()
        return trainer

    return run


def lr(trainer):
    return trainer.optimizer.param_groups[0]["lr"]


def test_training_saves_a_checkpoint(train, tmp_path):
    train(n_epochs=1)

    assert (tmp_path / "last_checkpoint.pytorch").is_file()


@pytest.mark.parametrize("schedule", ["Warmup", "WarmupPoly"])
def test_warmup_starts_below_the_initial_learning_rate(train, schedule):
    trainer = train(n_epochs=1, lr=schedule, warmEpochs=2, start_epoch=3)  # stops halfway through the warmup

    assert lr(trainer) < 1e-2


@pytest.mark.parametrize("schedule", ["ReduceLROnPlateau", "WarmupPlateau"])
def test_plateau_schedules_reduce_the_learning_rate(train, schedule):
    trainer = train(n_epochs=12, lr=schedule, factor=0.5, patience=1)  # WarmupPlateau waits for epoch 10

    assert lr(trainer) == pytest.approx(0.5e-2)


def test_checkpoint_loads_with_weights_only(train, tmp_path):
    import torch

    train(n_epochs=1)

    state = torch.load(tmp_path / "last_checkpoint.pytorch", weights_only=True)

    assert state["scheduler_state_dict"]["schedulers"][0]["update_steps"] > 0


def test_trained_model_loads_for_prediction(train, model_config):
    from deepanedet.utils import load_model

    train(n_epochs=1)

    _, epoch = load_model(model_config, last=True)

    assert epoch == 1


def test_training_without_scheduler_saves_a_checkpoint(train, tmp_path):
    train(n_epochs=1, lr=None)

    assert (tmp_path / "last_checkpoint.pytorch").is_file()


def test_resumed_training_continues_the_warmup(train, model_config, tmp_path):
    warmup = {"lr": "WarmupPoly", "warmEpochs": 3, "start_epoch": 4}
    uninterrupted = lr(train(n_epochs=2, **warmup, model_file=str(tmp_path / "other" / "last_checkpoint.pytorch")))

    train(n_epochs=1, **warmup)
    resumed = lr(train(n_epochs=2, **warmup))

    assert resumed == pytest.approx(uninterrupted)


def test_training_on_cpu_does_not_ask_for_cuda(train, recwarn):
    train(n_epochs=1, mixed_precision=True)

    assert [str(w.message) for w in recwarn if "CUDA" in str(w.message)] == []


def test_mixed_precision_can_be_turned_off(train):
    trainer = train(n_epochs=1, mixed_precision=False)

    assert not trainer.use_amp and not trainer.mixed_precision.is_enabled()
