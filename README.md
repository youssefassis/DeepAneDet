> **Note:** This repository is a copy of the original code hosted at [gitlab.inria.fr/yassis/DeepAneDet](https://gitlab.inria.fr/yassis/DeepAneDet), developed during my PhD at Inria/LORIA.

# DeepAneDet

Detection of intracranial aneurysms in 3D Time-of-Flight MRA as an object detection problem: a 3D single-stage,
anchor-free network predicts each aneurysm as a sphere. This is the code and the annotations of the paper
["Intracranial Aneurysm Detection: An object detection perspective"](https://link.to.paper/).

<details>
<summary>Abstract</summary>

Intracranial aneurysm detection from 3D Time-Of-Flight Magnetic Resonance Angiography images is a problem of
increasing clinical importance. Recently, a streak of methods have shown promising performance by using segmentation
neural networks. However, these methods may be less relevant in a clinical settings where diagnostic decisions rely
on detecting objects rather than their segmentation. We introduce a 3D single-stage object detection method tailored
for small object detection such as aneurysms. Our anchor-free method incorporates fast data annotation, adapted data
sampling and generation to address class imbalance problem, and spherical representations for improved object
detection. A comprehensive evaluation was conducted, comparing our method with the state-of-the-art SCPM-Net[1],
nnDetection[2] and nnUNet[3] baselines, using two datasets comprising 402 subjects, including one public dataset [4].
The evaluation used adapted object detection metrics. Our method exhibited comparable or superior performance, with
an average precision of 78.96%, sensitivity of 86.78%, and 0.53 false positives per case. Our method significantly
reduces the detection complexity compared to existing methods, and highlights the advantages of object detection
over segmentation-based approaches for aneurysm detection. It also holds potential for application to other small
object detection problems.
</details>

## Installation
The project uses [uv](https://docs.astral.sh/uv/) and Python 3.10+:
```
git clone https://github.com/youssefassis/DeepAneDet.git && cd DeepAneDet
uv sync --extra cu126    # NVIDIA GPU (CUDA 12.6)
uv sync --extra cpu      # or CPU only
```
Commands run from the repository folder with `uv run`.

**Docker:** `scripts/build_docker.sh` builds the image (`scripts/build_docker.sh cpu` for CPU only), and
`scripts/run_docker.sh` starts JupyterLab on http://localhost:5000 as your user, with your home folder mounted at the
same path (the token is shown by `docker logs deepanedet`). Inside the container, run the commands without `uv run`.

## Data
Each patient is a folder, either `P0001`-like or `sub-XXX/ses-YYYYMMDD` as in the public dataset:
```
Data/
  P0001/
    volume.nii.gz    TOF-MRA scan
    aneurysms.csv    aneurysms: two points (x, y, z columns, in mm) per aneurysm, see Annotations
    config.json      {"init volume": "volume.nii.gz", "pts aneurysm": "aneurysms.csv"}
    noskull.nii.gz   skull-stripped scan (written by remove_skull.py)
    points.csv       negative patch centers (written by extract_points.py)
```
Patients without aneurysm have no `aneurysms.csv`. A split file lists the training, validation and testing patient
folders, as in `reproducibility/fold1.json`.

## Usage
```
uv run python scripts/remove_skull.py Data/P*                                       # 1. skull stripping
uv run python scripts/extract_points.py Data/P*                                     # 2. negative patch centers
uv run python scripts/prepare.py Data Work reproducibility/fold1.json --name Fold1  # 3. training folder
uv run python scripts/train.py Work/Fold1                                           # 4. training
uv run python scripts/predict.py Work/Fold1                                         # 5. prediction
uv run python scripts/evaluate.py Work/Fold1/Predictions/<n>_epochs/Validation_0.01/None_detections_per_patch
```
- `prepare.py` writes `Work/Fold1/ndl_config.json`; the other training settings (patch size, augmentation,
  optimizer...) are set in the script.
- Training saves `last_checkpoint.pytorch` and TensorBoard logs in the training folder.
- Prediction runs on the validation and testing patients of the split, and writes one JSON file per patient
  (`P0001.json`, `sub-013_ses-20101220.json`): each detection has a center (x, y, z) and a radius in mm, and a
  confidence between 0 and 1.
- Evaluation matches the detections, from the most to the least confident, to the aneurysm they overlap most
  (IoU ≥ 0.1), and writes a CSV table of the detections and a summary figure next to the predictions.

## Annotations
Each aneurysm is approximated by a sphere defined by two points: the center of its neck (F1) and its dome (F2).

<img src="docs/annotation.png" alt="Two points F1 and F2 define the sphere (red) that approximates the aneurysm">

## Reproducing the paper
- `reproducibility/annotations`: annotations of the public dataset [4], one JSON file per subject. Write them as the
  `aneurysms.csv` of each patient folder with:
  ```
  uv run python scripts/convert_annotations.py reproducibility/annotations Data
  ```
- `reproducibility/fold1.json` to `fold5.json`: the 5-fold cross-validation splits. Replace their
  `/path/to/directory` prefix with your data folder.
- `baselines`: the nnU-Net [3] and nnDetection [2] baselines (see its README).

## References
[1] Xiangde et al., SCPM-Net: An anchor-free 3D lung nodule detection network using sphere representation and
center points matching. Medical Image Analysis 75, 102287 (2022).

[2] Baumgartner et al., nnDetection: A self-configuring method for medical object detection. MICCAI, pp. 530–539
(2021).

[3] Isensee et al., nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature
Methods 18(2), pp. 203–211 (2021).

[4] Di Noto et al., Towards automated brain aneurysm detection in TOF-MRA: open data, weak labels, and anatomical
knowledge. Neuroinformatics, pp. 1–14 (2022).

## Acknowledgements
This work was funded by Region Grand-Est and CHRU University Hospital of Nancy, France.
