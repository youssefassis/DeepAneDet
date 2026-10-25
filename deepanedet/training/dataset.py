import os
import random

import numpy as np
import torch

from deepanedet.utils import detection_grid_size, get_logger

from deepanedet.data.generators import generate_transforms, get_patches
from deepanedet.data.augmentation import random_flip
from deepanedet.volume.patch import get_patch_and_aneurysms
from deepanedet.training.batch_sampler import BalancedBatchSampler

logger = get_logger("Dataset")


def get_dataloaders(train_db, valid_db, cfg):
    """
    Returns the training and validation data loaders (None without patients) and their numbers of batches.
    cfg: training configuration (see configs/train.yaml), providing data.patch_size, data.patch_shape, model.num_levels,
    augmentation, batch_size, validation_batch_size (with valid_db), balanced_batch,
    negative_patches, drop_last and num_workers.
    Training batches hold a fixed share of negative patches with balanced_batch, else are randomly drawn.
    """
    workers = min(cfg["num_workers"], os.cpu_count() or 1)
    augmentation = cfg["augmentation"]
    scales = [detection_grid_size(cfg["data"]["patch_shape"][0], cfg["model"]["num_levels"])]

    def make_dataset(patches):
        return Dataset(
            patches=patches,
            size=list(cfg["data"]["patch_size"]),
            dim=list(cfg["data"]["patch_shape"]),
            negative=augmentation["negative"],
            positive=augmentation["positive"],
            flip=augmentation["flip"],
            scales=scales,
        )

    train_generator, train_iterations = None, 0
    valid_generator, valid_iterations = None, 0

    if train_db is not None:
        train_patches, train_iterations = get_patches(
            pat_db=train_db, dup_ane=augmentation["positive"]["duplicates"], batch_size=cfg["batch_size"]
        )
        train_dataset = make_dataset(train_patches)
        if cfg["balanced_batch"]:
            sampler = BalancedBatchSampler(
                patches=train_dataset.patches,
                batch_size=cfg["batch_size"],
                percentage_neg_patches=cfg["negative_patches"],
                drop_last=cfg["drop_last"],
            )
            train_iterations = sampler.get_n_batches()
            train_generator = torch.utils.data.DataLoader(
                dataset=train_dataset, num_workers=workers, pin_memory=True, batch_sampler=sampler
            )
        else:
            train_generator = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=workers, pin_memory=True
            )
        logger.info(
            f"Training DataLoader : ({len(train_patches)} patches; Balanced batch : {cfg['balanced_batch']}; batch size = {cfg['batch_size']}; workers = {workers})"
        )

    if valid_db is not None:
        valid_patches, valid_iterations = get_patches(
            pat_db=valid_db, dup_ane=augmentation["positive"]["duplicates"], batch_size=cfg["validation_batch_size"]
        )
        valid_generator = torch.utils.data.DataLoader(
            dataset=make_dataset(valid_patches),
            batch_size=cfg["validation_batch_size"],
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )
        logger.info(
            f"Validation DataLoader : ({len(valid_patches)} patches; batch size = {cfg['validation_batch_size']}; workers = {workers})"
        )
    return {"train": train_generator, "valid": valid_generator}, {"train": train_iterations, "valid": valid_iterations}


class Dataset(torch.utils.data.Dataset):
    """
    This is a PyTorch Dataset class that defines the data loading pipeline for the neural network.
    The dataset is assumed to consist of 3D images and their corresponding aneurysms (as markup points).
    The class takes several arguments such as the patches volume (i.e., the images and their aneurysms),
    patch size, dimension, transformations (translation, rotation, distorsion, scaling, flipping) for
    positive and negative patches, and the scales size as a list (e.g. [12]).
    """

    def __init__(self, patches, size, dim, negative, positive, flip, scales):
        """negative, positive: augmentation amplitudes (shift, rotation, distortion, scaling); flip: probability, axes"""
        random.shuffle(patches)
        self.patches = patches
        self.size = size
        self.dim = dim
        self.neg_trans, self.neg_rot = negative["shift"], negative["rotation"]
        self.neg_disp, self.neg_scal = negative["distortion"], negative["scaling"]
        self.pos_trans, self.pos_rot = positive["shift"], positive["rotation"]
        self.pos_disp, self.pos_scal = positive["distortion"], positive["scaling"]
        self.flip_prob, self.flip_axis = flip["probability"], list(flip["axes"])

        self.scales = scales

        logger.info(f"Patch size = {[round(elem, 2) for elem in self.size]}")
        logger.info(f"Patch dimension = {self.dim}")

        logger.info(f"Scales: {self.scales}")

        logger.info(
            f"Positive patches: Scaling: {self.pos_scal}; Translation: {self.pos_trans}; Rotation: {self.pos_rot}; Distorision: {self.pos_disp}; Flip axis: {self.flip_axis}; Flip probability {self.flip_prob}"
        )
        logger.info(
            f"Negative patches: Scaling: {self.neg_scal}; Translation: {self.neg_trans}; Rotation: {self.neg_rot}; Distorision: {self.neg_disp}; Flip axis: {self.flip_axis}; Flip probability {self.flip_prob}"
        )

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        """
        This method first retrieves the patch at the given index. It then determines which set
        of transformation parameters to use based on the value of p['status']. If p['status']
        is True, it uses the positive transformation parameters; otherwise, it uses the negative
        transformation parameters.
        An infinite loop is used to make sure that the augmented aneurysm points are fullt included
        inside the patch volume and maintains a 1.2mm distance from the borders (see get_patch_and_aneurysms).
        If the points are not fully covered inside the patch, new random transformations are
        generated using the generate_transforms.
        Once it has generated a fully-covered aneurysm, the method applies a random flip to the
        volume and associated aneurysms using the random_flip function.
        Finally, a list of targets for each scale is generated using the get_truth_from_aneurysms
        function, passing in the associated aneurysms, scales, and the patch dimension (self.dim[0]).
        It then converts each set of targets to a PyTorch tensor.
        """
        p = self.patches[index]

        affine, disp = None, None
        if p["status"]:
            trans, rot, scal, dispc = self.pos_trans, self.pos_rot, self.pos_scal, self.pos_disp
        else:
            trans, scal, rot, dispc = self.neg_trans, self.neg_scal, self.neg_rot, self.neg_disp
        repeat_patch_gen = True
        while repeat_patch_gen:
            affine, disp = generate_transforms(trans=trans, scal=scal, rot=rot, center=p["point"], disp=dispc)
            patch_data = get_patch_and_aneurysms(
                vol=p["data"],
                vox2met=p["vox2met"],
                center=p["point"],
                size=self.size,
                dim=self.dim,
                aneurysms=p["aneurysms"],
                affine=affine,
                disp=disp,
            )
            v, aneurysms, fully_covered = patch_data["volume"], patch_data["aneurysms"], patch_data["fully_covered"]
            repeat_patch_gen = not fully_covered
        v, aneurysms = random_flip(
            data=[v, aneurysms], axes=self.flip_axis, patch_shape=self.dim, flip_probability=self.flip_prob
        )

        targets = get_spheres_from_aneurysms_points(points=aneurysms / self.dim[0], scales=self.scales)
        targets = [[torch.from_numpy(scale[0]), torch.from_numpy(scale[1])] for scale in targets]

        return {"volume": torch.from_numpy(v[np.newaxis]).float(), "truth": targets}


def get_spheres_from_aneurysms_points(points, scales):
    """
    This is a function that takes in a set of points (representing the two ends of an aneurysm),
    a list of scales. It then creates a set of empty target tensors for each scale in the list
    of scales, with dimensions equal to the input volume dimensions scaled by the corresponding
    scale. For each pair of points, the function calculates the center point between them, scales
    it to the appropriate size for each scale, and uses the resulting coordinates to set the
    appropriate values in the corresponding target tensors. Specifically, it sets the confidence
    value to 1 (indicating the presence of an aneurysm in that cell), the center point to the
    scaled coordinates of the center point, and diameter to the scaled difference between
    the center point and one of the end points. The function returns the list of target tensors
    for each scale.
    """
    targets = [[np.zeros((S, S, S, 1), dtype="float32"), np.zeros((S, S, S, 4), dtype="float32")] for S in scales]
    for idx in range(0, points.shape[0], 2):
        p1, p2 = points[idx], points[idx + 1]
        center = (p1 + p2) / 2
        radius = np.linalg.norm(p1 - p2) / 2  # *1.33
        x, y, z = center
        for idx, S in enumerate(scales):
            i, j, k = int(S * x), int(S * y), int(S * z)  # which cell in the volume/image
            x_cell, y_cell, z_cell = (
                S * x - i,
                S * y - j,
                S * z - k,
            )  # each one is between [0, 1] center coords relative to the cell
            radius_cell = radius * S
            targets[idx][0][i, j, k, 0] = 1
            targets[idx][1][i, j, k, 0:] = np.array([x_cell, y_cell, z_cell, radius_cell])
    return targets
