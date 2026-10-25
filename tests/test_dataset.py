from deepanedet.training.dataset import get_dataloaders

from conftest import TINY_TRAINING, compose_config


def test_dataloaders_include_validation(patient_db, tmp_path):
    cfg = compose_config("train", [*TINY_TRAINING, f"data_dir={tmp_path}", "+validation_batch_size=2"])

    loaders, _ = get_dataloaders(train_db=patient_db[:1], valid_db=patient_db[1:], cfg=cfg)

    batch = next(iter(loaders["valid"]))

    assert batch["volume"].shape[1:] == (1, 16, 16, 16)
