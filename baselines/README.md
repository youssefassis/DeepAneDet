# Baselines

The nnU-Net and nnDetection baselines of the paper, as patches on the upstream versions they were run with. They are
kept for reproducibility and are not used by DeepAneDet itself.

| Patch | Upstream | Changes for the paper |
|---|---|---|
| `nnunet-v1.7.0.patch` | [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) `v1.7.0` | `inference/segmentation_export.py` saves the foreground probability instead of the label (used as detection confidence; the original is kept as `segmentation_export_org.py`); `nnUNetTrainerV2_Loss_Dice.py` adds a Dice + CE trainer without smoothing; small changes to `inference/predict.py`, `predict_simple.py`, `run/run_training.py` and the pretrained model scripts; `setup.py` leaves the dependencies to the environment |
| `nndetection-1044ace5.patch` | [nnDetection](https://github.com/MIC-DKFZ/nnDetection) commit `1044ace5` | `nndet` is unchanged; a Docker image with Jupyter (`Dockerfile`, `buildDocker.sh`, `runDocker.sh`), `${env:...}` paths in `scripts/`, and an extra `scripts/predict_old.py` |

To get the code that was run, check out each upstream version next to DeepAneDet and apply its patch:
```
git clone https://github.com/MIC-DKFZ/nnUNet && git -C nnUNet checkout v1.7.0
git -C nnUNet apply ../DeepAneDet/baselines/nnunet-v1.7.0.patch
git clone https://github.com/MIC-DKFZ/nnDetection && git -C nnDetection checkout 1044ace5
git -C nnDetection apply ../DeepAneDet/baselines/nndetection-1044ace5.patch
```
Then install and run them as described in their own documentation.

`prepare_nnunet_data.ipynb` and `prepare_nndetection_data.ipynb` convert the data to the formats these baselines expect.
