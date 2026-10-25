import pytest

from hydra.utils import instantiate

from deepanedet.training.dataset import get_dataloaders
from deepanedet.training.losses import get_loss_criterion
from deepanedet.training.trainer import create_trainer
from deepanedet.utils import create_lr_scheduler, get_model

from conftest import TINY_TRAINING, compose_config


@pytest.fixture
def train(patient_db, tmp_path):
    """
    Trains (or resumes training) the tiny model up to n_epochs, with Hydra overrides of the training config, in
    train_dir (tmp_path by default); returns its trainer.
    """

    def run(n_epochs, *overrides, train_dir=tmp_path):
        cfg = compose_config("train", [*TINY_TRAINING, f"data_dir={tmp_path}", f"epochs={n_epochs}", *overrides])
        loaders, iterations = get_dataloaders(patient_db, None, cfg)
        model = get_model(cfg.model)
        optimizer = instantiate(cfg.optimizer, params=model.parameters(), _convert_="all")
        warmup_epochs = cfg.scheduler.warmup_epochs
        scheduler = create_lr_scheduler(optimizer, cfg.scheduler, cfg.epochs, iterations["train"] * warmup_epochs)
        trainer = create_trainer(
            str(train_dir),
            "cpu",
            model,
            optimizer,
            scheduler,
            get_loss_criterion(cfg.loss),
            loaders,
            iterations,
            max_num_epochs=cfg.epochs,
            save_every=cfg.save_every,
            warmup_epochs=warmup_epochs,
            mixed_precision=cfg.mixed_precision,
        )
        trainer.fit()
        return trainer

    return run


def lr(trainer):
    return trainer.optimizer.param_groups[0]["lr"]


def test_training_saves_a_checkpoint(train, tmp_path):
    train(1)

    assert (tmp_path / "last_checkpoint.pytorch").is_file()


@pytest.mark.parametrize("schedule", ["Warmup", "WarmupPoly"])
def test_warmup_starts_below_the_initial_learning_rate(train, schedule):
    trainer = train(
        1, f"scheduler.name={schedule}", "scheduler.warmup_epochs=2", "scheduler.start_epoch=3"
    )  # stops halfway through the warmup

    assert lr(trainer) < 1e-2


@pytest.mark.parametrize("schedule", ["ReduceLROnPlateau", "WarmupPlateau"])
def test_plateau_schedules_reduce_the_learning_rate(train, schedule):
    trainer = train(
        12, f"scheduler.name={schedule}", "+scheduler.factor=0.5", "+scheduler.patience=1"
    )  # WarmupPlateau waits for epoch 10

    assert lr(trainer) == pytest.approx(0.5e-2)


def test_checkpoint_loads_with_weights_only(train, tmp_path):
    import torch

    train(1)

    state = torch.load(tmp_path / "last_checkpoint.pytorch", weights_only=True)

    assert state["scheduler_state_dict"]["schedulers"][0]["update_steps"] > 0


def test_trained_model_loads_for_prediction(train, tmp_path):
    from deepanedet.utils import load_model

    train(1)

    _, epoch = load_model(
        compose_config("train", [f"data_dir={tmp_path}", *TINY_TRAINING]).model, tmp_path / "last_checkpoint.pytorch"
    )

    assert epoch == 1


def test_training_without_scheduler_saves_a_checkpoint(train, tmp_path):
    train(1, "scheduler.name=null")

    assert (tmp_path / "last_checkpoint.pytorch").is_file()


def test_resumed_training_continues_the_warmup(train, tmp_path):
    warmup = ["scheduler.name=WarmupPoly", "scheduler.warmup_epochs=3", "scheduler.start_epoch=4"]
    (tmp_path / "other").mkdir()
    uninterrupted = lr(train(2, *warmup, train_dir=tmp_path / "other"))

    train(1, *warmup)
    resumed = lr(train(2, *warmup))

    assert resumed == pytest.approx(uninterrupted)


def test_training_on_cpu_does_not_ask_for_cuda(train, recwarn):
    train(1, "mixed_precision=true")

    assert [str(w.message) for w in recwarn if "CUDA" in str(w.message)] == []


def test_mixed_precision_can_be_turned_off(train):
    trainer = train(1, "mixed_precision=false")

    assert not trainer.use_amp and not trainer.mixed_precision.is_enabled()
