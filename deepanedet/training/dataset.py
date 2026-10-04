import torch
import random
import numpy as np
from deepanedet.utils import get_logger

from deepanedet.data.generators import generateTransforms, getPatches
from deepanedet.data.augmentation import randomFlip
from deepanedet.volume.patch import getPatchAndAneurysms
from deepanedet.training.batch_sampler import BalancedBatchSampler

logger = get_logger("Dataset")


def getDataloaders(train_db, valid_db, config, workers):
    """
    This function returns PyTorch data loaders for training and validation sets,
    given the training and validation databases, a configuration dictionary,
    and the number of worker processes to use.
    The function first checks whether a training/validation database is provided.
    If so, it calls the getPatches() function to obtain patches from the database
    with the specified batch size and positive patch duplicates. Then, it creates
    a Dataset object with the obtained patches and the other configuration parameters,
    and creates a PyTorch data loader using either Balanced Batch Sampler or a standard
    random batch sampler.
    """
    train_generator, train_iterations = None, 0
    valid_generator, valid_iterations = None, 0

    balancedbatch = config["balancedbatch"] if "balancedbatch" in config else False
    scales = config["scales"]  # if 'scales' in config else None
    anchors = config["anchors"] if "anchors" in config else None

    if train_db is not None:
        train_patches, train_iterations = getPatches(
            pat_db=train_db, dup_ane=config["positive duplicates"], batch_size=config["batch_size"]
        )

        train_dataset = Dataset(
            patches=train_patches,
            size=[x * y for x, y in zip(config["patch_size"], config["patch_shape"])],
            dim=config["patch_shape"],
            neg_trans=config["negative sample shift"],
            neg_rot=config["negative sample rotation"],
            neg_disp=config["negative sample distortion"],
            neg_scal=config["negative sample scaling"],
            pos_trans=config["positive sample shift"],
            pos_rot=config["positive sample rotation"],
            pos_disp=config["positive sample distortion"],
            pos_scal=config["positive sample scaling"],
            flip_prob=config["flip probability"],
            flip_axes=config["flip axes"],
            params=config["num parameters"],
            scales=scales,
            anchors=anchors,
        )

        if balancedbatch:
            sampler = BalancedBatchSampler(
                patches=train_dataset.patches,
                batch_size=config["batch_size"],
                nb_negative_samples_per_patient=config["nb neg patches per patient"],
                percentage_neg_patches=config["percentage of negative patches"],
                drop_last=config["drop_last"],
            )
            train_iterations = sampler.get_n_batches()
            train_generator = torch.utils.data.DataLoader(
                dataset=train_dataset, num_workers=workers, pin_memory=True, batch_sampler=sampler
            )
        else:
            train_generator = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=workers,
                pin_memory=True,
            )

        logger.info(
            f"Training DataLoader : ({len(train_patches)} patches; Balanced batch : {balancedbatch}; batch size = {config['batch_size']}; shuffle = {True}; workers = {workers})"
        )

    if valid_db is not None:
        valid_patches, valid_iterations = getPatches(
            pat_db=valid_db, dup_ane=config["positive duplicates"], batch_size=config["validation_batch_size"]
        )
        valid_dataset = Dataset(
            patches=valid_patches,
            size=[x * y for x, y in zip(config["patch_size"], config["patch_shape"])],
            dim=config["patch_shape"],
            neg_trans=config["negative sample shift"],
            neg_rot=config["negative sample rotation"],
            neg_disp=config["negative sample distortion"],
            neg_scal=config["negative sample scaling"],
            pos_trans=config["positive sample shift"],
            pos_rot=config["positive sample rotation"],
            pos_disp=config["positive sample distortion"],
            pos_scal=config["positive sample scaling"],
            flip_prob=config["flip probability"],
            flip_axes=config["flip axes"],
            params=config["num parameters"],
            scales=scales,
            anchors=anchors,
        )
        valid_generator = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=config["validation_batch_size"],
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )

        logger.info(
            f"Validation DataLoader : ({len(valid_patches)} patches; Balanced batch : {balancedbatch}; batch size = {config['validation_batch_size']}; shuffle = False; workers = {workers})"
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

    def __init__(
        self,
        patches,
        size,
        dim,
        neg_trans,
        neg_rot,
        neg_disp,
        neg_scal,
        pos_trans,
        pos_rot,
        pos_disp,
        pos_scal,
        flip_prob,
        flip_axes,
        params,
        scales,
        anchors,
    ):
        random.shuffle(patches)
        self.patches = patches
        self.size = size
        self.dim = dim
        self.neg_trans, self.neg_rot, self.neg_disp, self.neg_scal = neg_trans, neg_rot, neg_disp, neg_scal
        self.pos_trans, self.pos_rot, self.pos_disp, self.pos_scal = pos_trans, pos_rot, pos_disp, pos_scal
        self.flip_prob, self.flip_axis = flip_prob, flip_axes

        self.scales = scales
        self.anchors = anchors
        self.nb_parameters = params

        logger.info(f"Patch size = {[round(elem, 2) for elem in self.size]}")
        logger.info(f"Patch dimension = {self.dim}")

        logger.info(f"Scales: {self.scales}")
        logger.info(f"Anchors: {self.anchors}")

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
        inside the patch volume and maintains a 1.2mm distance from the borders (see getPatchAndAneurysms).
        If the points are not fully covered inside the patch, new random transformations are
        generated using the generateTransforms.
        Once it has generated a fully-covered aneurysm, the method applies a random flip to the
        volume and associated aneurysms using the randomFlip function.
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
            affine, disp = generateTransforms(trans=trans, scal=scal, rot=rot, center=p["point"], disp=dispc)
            patch_data = getPatchAndAneurysms(
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
        v, aneurysms = randomFlip(
            data=[v, aneurysms], axes=self.flip_axis, patch_shape=self.dim, flip_probability=self.flip_prob
        )

        if self.anchors is None:
            if self.nb_parameters == 4:  # bounding spheres
                targets = get_spheres_from_aneurysms_points(points=aneurysms / self.dim[0], scales=self.scales)
            elif self.nb_parameters == 6:  # bounding boxes
                targets = get_boxes_from_aneurysms_points(points=aneurysms, scales=self.scales, dim=self.dim[0])

        else:
            targets = get_spheres_from_aneurysms_points_with_anchors(
                points=aneurysms / self.dim[0], scales=self.scales, anchors=self.anchors
            )

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


def get_spheres_from_aneurysms_points_with_anchors(points, scales, anchors, ignore_iou_thresh=0.5):
    def get_ious(spheres1, spheres2):
        spheres1 = np.array(spheres1)
        spheres2 = np.array(spheres2)
        intersection = np.minimum(spheres1, spheres2)
        union = np.maximum(spheres1, spheres2)
        return intersection / union

    targets = [
        [
            np.zeros((len(anchors), S, S, S, 1), dtype="float32"),  # confidence
            np.zeros((len(anchors), S, S, S, 4), dtype="float32"),
        ]  # parameters (x, y, z, radius)
        for S in scales
    ]

    num_anchors_per_scale = len(anchors) // len(scales)
    for idx in range(0, points.shape[0], 2):  # for each aneurysm/sphere inside the patch volume
        p1, p2 = points[idx], points[idx + 1]
        center = (p1 + p2) / 2
        radius = np.linalg.norm(p1 - p2) / 2
        x, y, z = center

        iou_anchors = get_ious(radius, anchors)
        anchor_indices = (-iou_anchors).argsort()

        has_anchor = [False] * len(
            anchors
        )  # make sure at the end that there is a bounding sphere for each of three scales
        for anchor_idx in anchor_indices:
            scale_idx = anchor_idx // num_anchors_per_scale
            anchor_on_scale = anchor_idx % num_anchors_per_scale
            S = scales[scale_idx]  # anchors in that scale
            i, j, k = int(S * x), int(S * y), int(S * z)  # which cell in the patch
            anchor_taken = targets[scale_idx][0][anchor_on_scale, i, j, k, 0]

            if not has_anchor[scale_idx] and not anchor_taken:
                x_cell, y_cell, z_cell = (
                    S * x - i,
                    S * y - j,
                    S * z - k,
                )  # each one is between [0, 1] center coords relative to the cell
                radius_cell = radius * S
                targets[scale_idx][0][anchor_on_scale, i, j, k, 0] = (
                    1  # there is an object here, so truth confidence = 1
                )
                targets[scale_idx][1][anchor_on_scale, i, j, k, 0:] = np.array([x_cell, y_cell, z_cell, radius_cell])
                has_anchor[scale_idx] = True

            elif not anchor_taken and iou_anchors[anchor_idx] > ignore_iou_thresh:
                targets[scale_idx][0][anchor_on_scale, i, j, k, 0] = -1  # ignore detection
    return targets


def get_boxes_from_aneurysms_points(points, scales, dim=96):
    targets = [[np.zeros((S, S, S, 1), dtype="float32"), np.zeros((S, S, S, 6), dtype="float32")] for S in scales]
    for idx in range(0, points.shape[0], 2):
        p1, p2 = points[idx], points[idx + 1]
        center = ((p1 + p2) / 2) / dim
        x, y, z = center
        for idx, S in enumerate(scales):
            i, j, k = int(S * x), int(S * y), int(S * z)  # which cell in the volume
            x_cell, y_cell, z_cell = (
                S * x - i,
                S * y - j,
                S * z - k,
            )  # each one is between [0, 1] center coords relative to the cell

            length = (np.linalg.norm(p1[0] - p2[0]) / 2) / dim
            width = (np.linalg.norm(p1[1] - p2[1]) / 2) / dim
            depth = (np.linalg.norm(p1[2] - p2[2]) / 2) / dim
            length_cell, width_cell, depth_cell = (
                length * S,
                width * S,
                depth * S,
            )  # can be greater than 1 since it's relative to cell

            targets[idx][0][i, j, k, 0] = 1  # there is an object here, so truth confidence = 1
            targets[idx][1][i, j, k, 0:] = np.array([x_cell, y_cell, z_cell, length_cell, width_cell, depth_cell])
    return targets
