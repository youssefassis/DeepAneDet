from dataset import getDataloaders


def test_dataloaders_include_validation(patient_db, training_config):
    loaders, _ = getDataloaders(train_db=patient_db[:1], valid_db=patient_db[1:], config=training_config, workers=0)

    batch = next(iter(loaders["valid"]))

    assert batch["volume"].shape[1:] == (1, 16, 16, 16)
