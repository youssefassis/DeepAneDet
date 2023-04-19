# Aneurysm Detection using Spherical Representation
This repository contains the official software code and annotations related to the paper ["Intracranial Aneurysm Detection using Spherical Representation"](https://link.to.paper/). The paper was published in IEEE Transactions on Medical Imaging 2023.

# Abstract
Intracranial aneurysms detection from 3D Time-Of-Flight Magnetic Resonance Angiography (TOF-MRA) images is a problem of increasing clinical importance. This paper proposes a 3D object detection method designed specifically for the detection of small 3D ball-shaped objects such as aneurysms, using spherical representation. The proposed method is inspired by YOLO architecture and is based on fast data annotation and adapted data sampling and generation strategies.

We compare our method to state-of-the-art nnDetection[1] and nnUnet[2] methods using two datasets comprising 402 patients, including one public dataset[3], and using more adapted evaluation metrics. Our approach significantly reduces the detection complexity while achieving comparable or even superior performance, with an average precision of 78.96%, and a sensitivity of 86.78% associated with 0.53 false positives per case.

# Description
The project is organized into two main directories: "Data_dir" and "Work_dir". The "Data_dir" contains data related to patients (P0001, P0002, etc.) and a working directory called "Work_dir" that stores training samples (Train001, Train002, etc.) for custom training.

1. Within each patient's folder in "Data_dir", there are several files such as "volume.nii.gz", which contains the 3D MRA image, "noskull.nii.gz", which contains the brain (without skull) volume, "aneurysm.csv", which lists the coordinates of points defining each aneurysm, and "points.csv", which lists the coordinates of candidate negative patches used for training. All of these files are grouped together under a "config.json" file. See an example in reproductibity/Data/P0001.

2. In the "Work_dir", there is a directory called "Train001" that contains the configuration of a customized training (called "ndl_config.json") and stores the checkpoint model called "last_checkpoint.pytorch". see an example in reproductibity/Work_dir/Train001.

# Usage
To use our code, follow the following steps:
## Preparation
1. Clone the repository by running the following commands:
    ```
    git clone https://gitlab.inria.fr/yassis/aneurysm-detection-spheres.git
    cd aneurysm-detection-spheres
    ```
2. Edit the "Docker" file by specifying the user_name, group_name, PYTHONPATH, and the configuration of Jupyterlab.
3. Create the Docker container by running the following commands:
    ```
    chmod +x buildDocker.sh
    ./buildDocker
    ```
4. Run the Docker container by running the following two commands:
    ```
    chmod +x runDocker.sh
    ./runDocker
    ```
5. Access to the Jupyterlab from your browser.
6. Prepare the data by generating the "noskull.nii.gz" and "points.csv" files for each patient in the dataset:
    ```
    python resources/scripts/removeSkull.py
    python resources/scripts/extractPoints.py
    ```

## Traning
1. Generate a customized training sample  (e.g. Train001 directory)  using the command:
    ```
    python prepare.py
    ```

2. Based on the generated configuration file in "Train001/ndl_config.json", start the training and validation phase by running the following two commands:
    ```
    chmd +x resources/scripts/train.py 
    ./resources/scripts/train.py path/to/Train001
    ```

## Inference
* Use the script "predict.py" to perform inference by running:
    ```
    chmd +x resources/scripts/predict.py
    ./resources/scripts/predict.py path/to/Train001
    ```
* The predictions are generated in "path/to/Train001/Predictions". For each patient, the predictions are saved in a separate JSON file. Each prediction includes the center coordinates (x, y, z) in mm, the radius in mm, and a confidence score as a percentage.


## Evaluation
- Evaluate the model using the script "evaluate.py" by running:
    ```
    chmd +x resources/scripts/evaluate.py
    ./resources/scripts/evaluate.py path/to/Train001
    ```
- After running the evaluation script, the results are generated as a PNG figure that summarizes the performance of the model on the test images. This figure can be found in the "path/to/Train001/Predictions" directory. 

# Citation
If you find this repository useful in your research, please consider citing:
```
Assis et al. Intracranial Aneurysm Detection using Spherical Representation (2023)
```
# References
[1] Isensee et al, nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation, Nature methods 18(2), 203–211 (2021).

[2] Baumgartner et al, nndetection: A self-configuring method for medical object detection. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 530–539. Springer (2021).

[3] Di Noto et al, Towards automated brain aneurysm detection in TOF-MRA: open data, weak labels, and anatomical knowledge. Neuroinformatics, pp. 1-14, (2022).

# Acknowledgements
This work was founded by Region Grand-Est, CHRU University hospital Nancy and Lorraine Univeristy in France.
# License
This project is licensed under the MIT License - see the LICENSE.md file for details.
