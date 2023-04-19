# Aneurysm-Detection-Using-Spherical-Representation
This repository provides the official software code and annotations of the paper **Intracranial Aneurysm Detection using Spherical Representation**. The paper was published in IEEE Transaction on Medical Imaging on 11 Juin 2023.

# Abstract
Intracranial aneurysms detection from 3D Time-Of-Flight Magnetic Resonance Angiography (TOF-MRA) images is a problem of increasing clinical importance. Recently, a streak of methods have shown promising performance by using 3D semantic segmentation neural networks with a patch-based approach. However, these methods may be less relevant in a clinical settings, where diagnostic decisions depend on detecting objects rather than their segmentation, and resulting in less adapted object detection evaluation metrics. In this paper, we propose a 3D object detection method designed specifically for the detection of small 3D ball-shaped objects such as aneurysms. 
Inspired by YOLO architecture, we propose an anchor-free method based on fast data annotation, and adapted data sampling and generation strategies to detect aneurysms using spherical representation. We compare our method to state-of-the-art nnDetection and nnUnet methods using two datasets comprising 402 patients, including one public dataset, and using more adapted evaluation metrics. Our approach significantly reduces the detection complexity while achieving comparable or even superior performance, with an average precision of 78.96%, and a sensitivity of 86.78% associated with 0.53 false positives per case.

If you find this repository useful in your work, please consider citing:

Paper citation

# Project Description

# Installation

## Data Preparation
## Training
## Inference
## Evaluation

# References
[1]
Assis et al. (2021). 
An efficient data strategy for the detection of brain aneurysms from mra with deep learning. 
Deep Generative Models, and Data Augmentation, Labelling, and Imperfections, pp. 226–234.

[2]
Di Noto et al. (2022). 
Towards automated brain aneurysm detection in TOF-MRA: open data, weak labels, and anatomical knowledge. 
Neuroinformatics, 1-14(3).

