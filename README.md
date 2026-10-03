# Intracranial Aneurysm Detection: An object detection perspective
This repository contains the official software code and annotations related to the paper ["Intracranial Aneurysm Detection: An object detection perspective"](https://link.to.paper/).

# Abstract
Intracranial aneurysm detection from 3D Time-Of-Flight Magnetic Resonance Angiography images is a problem of increasing clinical importance. Recently, a streak of methods have shown promising performance by using segmentation neural networks. However, these methods may be less relevant in a clinical settings where diagnostic decisions rely on detecting objects rather than their segmentation. We introduce a 3D single-stage object detection method tailored for small object detection such as aneurysms. Our anchor-free method incorporates fast data annotation, adapted data sampling and generation to address class imbalance problem, and spherical representations for improved object detection. A comprehensive evaluation was conducted, comparing our method with the state-of-the-art SCPM-Net[1], nnDetection[2] and nnUNet[3] baselines, using two datasets comprising 402 subjects, including one public dataset [4]. The evaluation used adapted object detection metrics. Our method exhibited comparable or superior performance, with an average precision of 78.96%, sensitivity of 86.78%, and 0.53 false positives per case. Our method significantly reduces the detection complexity compared to existing methods, and highlights the advantages of object detection over segmentation-based approaches for aneurysm detection. It also holds potential for application to other small object detection problems.

# Description
The project is organized into two main directories: "Data_dir" and "Work_dir". The "Data_dir" contains data related to patients (P0001, P0002, etc.) and a working directory called "Work_dir" that stores training samples (Train001, Train002, etc.) for custom training.

1. Within each patient's folder in "Data_dir", there are several files such as "volume.nii.gz", which contains the 3D MRA image, "noskull.nii.gz", which contains the brain (without skull) volume, "aneurysm.csv", which lists the coordinates of points defining each aneurysm, and "points.csv", which lists the coordinates of candidate negative patches used for training. All of these files are grouped together under a "config.json" file. See an example in reproductibity/Data/P0001.

2. In the "Work_dir", there is a directory called "Train001" that contains the configuration of a customized training (called "ndl_config.json") and stores the checkpoint model called "last_checkpoint.pytorch". see an example in reproductibity/Work_dir/Train001.

# Usage
To use our code, follow the following steps:
## Preparation
1. Clone the repository by running the following commands:
    ```
    git clone https://gitlab.inria.fr/yassis/DeepAneDet.git
    cd DeepAneDet
    ```
2. Build the Docker image (PyTorch for NVIDIA GPUs by default, or `scripts/build_docker.sh cpu`):
    ```
    scripts/build_docker.sh
    ```
3. Run the container, as your user and with your home directory mounted at the same path:
    ```
    scripts/run_docker.sh
    ```
4. Open JupyterLab at http://localhost:5000, with the token shown by `docker logs deepanedet`.
5. Outside Docker, install the project with [uv](https://docs.astral.sh/uv/): `uv sync --extra cu126` (NVIDIA GPU) or `uv sync --extra cpu`. Run the commands below from the repository root (inside Docker, without `uv run`).
6. Prepare the data by generating the "noskull.nii.gz" and "points.csv" files of the patient directories (`path/to/Data_dir/sub-*/ses-*` for session folders):
    ```
    uv run python scripts/remove_skull.py path/to/Data_dir/P*
    uv run python scripts/extract_points.py path/to/Data_dir/P*
    ```
    Negative patch centers are kept away from the aneurysms listed under "pts aneurysm" in each "config.json", or in the file given with `--truth-file` (e.g. `--truth-file aneurysms.csv`).

## Traning
1. Generate a customized training sample (e.g. the Fold1 directory) from a split file; the other training settings are set in `scripts/prepare.py`:
    ```
    uv run python scripts/prepare.py path/to/Data_dir path/to/Work_dir reproducibility/fold1.json --name Fold1
    ```

2. Based on the generated configuration file in "Train001/ndl_config.json", start the training and validation phase by running the following two commands:
    ```
    uv run python scripts/train.py path/to/Train001
    ```

## Inference
* Use the script "predict.py" to perform inference by running:
    ```
    uv run python scripts/predict.py path/to/Train001
    ```
* The predictions of the validation and testing patients of the split file are generated in "path/to/Train001/Predictions/<epochs>_epochs/...". For each patient, the predictions are saved in a separate JSON file named after the patient folder ("P0001.json"), or after the subject and session for session folders ("sub-013/ses-20101220" gives "sub-013_ses-20101220.json"). Each prediction includes the center coordinates (x, y, z) in mm, the radius in mm, and a confidence score between 0 and 1.


## Evaluation
- Evaluate the model using the script "evaluate.py" by running:
    ```
    uv run python scripts/evaluate.py path/to/Train001/Predictions/<epochs>_epochs/Validation_0.01/None_detections_per_patch
    ```
- The validation and testing patients of the split file are evaluated against their "truth file" (see "ndl_config.json"). Detections are matched from the most to the least confident, each to the not yet detected aneurysm it overlaps most (IoU >= 0.1); other detections are false positives.
- After running the evaluation script, the results are generated as a CSV table of the detections and a PNG figure that summarizes the performance of the model on the test images, both in the predictions directory.

## Known limitations
- Prediction tiles the volume with overlapping patches whose borders (1/6 of a patch) are ignored, and leaves out what does not fit a whole number of tiles: along each axis, up to one patch width (38.4 mm with the default 96 × 0.4 mm patches), split between both ends of the volume, is never scanned.
- During training, an aneurysm closer than 3 voxels to the border of a patch is not used as a target, so the patch is treated as background around it.
- The no-object confidence loss is weighted by the number of cells containing an aneurysm in the batch: batches without any aneurysm do not contribute to it.

## Tests
```
uv sync --extra cpu
uv run pytest
uv run ruff check .
```

# Annotations and Reproducibility
In order to overcome the limitations of voxel-wise annotation, weak annotation is used to annotate aneurysms. This involves approximating the shape of aneurysms using spheres. 

As illustrated in the figure below, to create these spheres, two points are used as reference: the center of the neck of the aneurysm (F1) and the dome of the aneurysm (F2). The sphere is then created to enclose the aneurysm using these two points as a guide.

<img
  src="docs/annotation.png"
  alt="Our adopted annotation"
  title=" Fast aneurysm annotation: 2 points (F1, F2) approximate the aneurysm with a sphere (red)">

To support the reproducibility of our paper, we provide access to the annotations used in the public dataset [4] at "reproducibility/annotations". Each subject file contains the ground truth annotation of aneurysms as two points with 3D coordinates. Moreover, to replicate the 5-fold cross-validation approach employed in our paper, the subjects used in each fold can be found at "reproducibility/fold?.json". The code of the nnU-Net [3] and nnDetection [2] baselines is in "baselines" (see its README for the upstream versions and the changes made for the paper).

To train or evaluate on these annotations, write them as the truth file ("aneurysms.csv", the "truth file" of "scripts/prepare.py") of each patient directory, laid out as `Data_dir/sub-XXX/ses-YYYYMMDD`:
```
uv run python scripts/convert_annotations.py reproducibility/annotations path/to/Data_dir
```
Patients without an annotation file have no aneurysm.

# References
[1] Xiangde et al., SCPM-Net: An anchor-free 3D lung nodule detection network using sphere representation and center points matching. Medical image analysis, 75 pp. 102287. Elsevier (2022)

[2] Baumgartner et al., nndetection: A self-configuring method for medical object detection. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 530–539. Springer (2021).

[3] Isensee et al., nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation, Nature methods 18(2), pp. 203–211 (2021).

[4] Di Noto et al., Towards automated brain aneurysm detection in TOF-MRA: open data, weak labels, and anatomical knowledge. Neuroinformatics, pp. 1-14 (2022).


# Acknowledgements
This work was funded by Region Grand-Est, CHRU University hospital of Nancy in France.
