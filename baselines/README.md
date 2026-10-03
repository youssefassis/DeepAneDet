# Baselines

Code used to run the nnU-Net and nnDetection baselines of the paper. It is kept here for reproducibility and is not used by DeepAneDet itself.

| Folder | Upstream | Changes for the paper |
|---|---|---|
| `nnunet` | [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) 1.7.0 (v1) | `inference/segmentation_export.py` saves the foreground probability instead of the label (used as detection confidence); `nnUNetTrainerV2_Loss_Dice.py` adds a Dice + CE trainer without smoothing |
| `nnDetection` | [nnDetection](https://github.com/MIC-DKFZ/nnDetection) commit `1044ace5` | `nndet` is unchanged; Docker setup, data and model paths in `scripts/`, and an extra `scripts/predict_old.py` |

`Prepare_nnUNet_data.ipynb` and `Prepare_nnDetection_data.ipynb` convert the data to the formats these baselines expect.
