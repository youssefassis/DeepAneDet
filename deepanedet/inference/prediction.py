import os
import math
import torch

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from deepanedet.data import io as dio
from deepanedet.helpers import (
    cells_to_spheres,
    non_max_suppression,
    spheres_to_metric,
    save_spheres,
    keep_only_intersected_spheres,
    flip_coords,
)


def set_affine_spacing(affine, spacing):
    """
    This function adjusts the spacing of a given affine transformation matrix.
    The function takes in two arguments:
        - affine, which is a 4x4 NumPy array representing the affine transformation matrix,
        - spacing, which is a list of three values representing the desired spacing in x, y, and z dimensions.
    """

    def get_spacing_from_affine(affine):
        RZS = affine[:3, :3]
        return np.sqrt(np.sum(RZS * RZS, axis=0))

    scale = np.divide(spacing, get_spacing_from_affine(affine))
    affine_transform = np.diag(np.ones(4))
    np.fill_diagonal(affine_transform, list(scale) + [1])
    return np.matmul(affine, affine_transform)


def sitk_resample_to_spacing(image, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear, default_value=0.0):
    """
    This function takes a SimpleITK image, and resamples it to a new spacing. The
    resampling is performed using an interpolation method specified by interpolator,
    and any pixels that are not part of the original image are assigned a value
    specified by default_value. The first voxels of both grids share their center, as
    set_affine_spacing assumes when mapping resampled coordinates back to the scanner space.
    """
    zoom_factor = np.divide(image.GetSpacing(), new_spacing)
    new_size = np.asarray(np.ceil(np.round(np.multiply(zoom_factor, image.GetSize()), decimals=5)), dtype=np.int16)
    reference_image = sitk_new_blank_image(
        size=new_size,
        spacing=new_spacing,
        direction=image.GetDirection(),
        origin=image.GetOrigin(),
        default_value=default_value,
    )
    return sitk_resample_to_image(image, reference_image, interpolator=interpolator, default_value=default_value)


def sitk_resample_to_image(
    image, reference_image, default_value=0.0, interpolator=sitk.sitkLinear, transform=None, output_pixel_type=None
):
    """
    This function resamples a SimpleITK image to match the size, spacing, direction,
    and origin of a reference image. It takes as input the image to be resampled,
    the reference_image to which the image is resampled, the default_value for
    out-of-bounds pixels, the interpolator used for resampling, the transform to be
    applied to the image, and the output_pixel_type. It returns the resampled image
    in the same format as the output_pixel_type.
    """
    if transform is None:
        transform = sitk.Transform()
        transform.SetIdentity()
    if output_pixel_type is None:
        output_pixel_type = image.GetPixelID()
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetTransform(transform)
    resample_filter.SetOutputPixelType(output_pixel_type)
    resample_filter.SetDefaultPixelValue(default_value)
    resample_filter.SetReferenceImage(reference_image)
    return resample_filter.Execute(image)


def sitk_new_blank_image(size, spacing, direction, origin, default_value=0.0):
    """
    This function creates a new blank SimpleITK image with specified size,
    spacing, direction, origin, and default value. The image data is initialized
    as a 3D numpy array with the given size, and all elements are set to the default
    value. The returned object is a SimpleITK image object, which can be used
    for various image processing tasks.
    """
    image = sitk.GetImageFromArray(np.ones(size, dtype=np.float64).T * default_value)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image


def resample_to_spacing(data, spacing, target_spacing, interpolation="linear", default_value=0.0):
    """
    The resample_to_spacing function takes in a 3D numpy array data, its original voxel
    spacing spacing, the target voxel spacing target_spacing, an interpolation method
    interpolation, and a default_value to fill any voxels that are outside the original
    image domain after resampling.
    The function first converts the data array into a SimpleITK image using the
    data_to_sitk_image function. Then it uses the sitk_resample_to_spacing function
    to resample the image to the target spacing, with the interpolation method specified
    by interpolation and with default_value used for values outside the original image
    domain. Finally, the resampled image is converted back to a numpy array using the
    sitk_image_to_data function and returned.
    """
    image = data_to_sitk_image(data, spacing=spacing)
    if interpolation == "linear":
        interpolator = sitk.sitkLinear
    elif interpolation == "nearest":
        interpolator = sitk.sitkNearestNeighbor
    else:
        raise ValueError(
            "'interpolation' must be either 'linear' or 'nearest'. '{}' is not recognized".format(interpolation)
        )
    resampled_image = sitk_resample_to_spacing(
        image, new_spacing=target_spacing, interpolator=interpolator, default_value=default_value
    )
    return sitk_image_to_data(resampled_image)


def data_to_sitk_image(data, spacing=(1.0, 1.0, 1.0)):
    """
    Returns the (x, y, z) volume data as a SimpleITK image of the given voxel spacing; SimpleITK arrays are indexed
    (z, y, x), hence the transpose.
    """
    image = sitk.GetImageFromArray(data.T)
    image.SetSpacing(np.asarray(spacing, dtype=np.float64))
    return image


def sitk_image_to_data(image):
    """Returns the pixel data of a SimpleITK image as an (x, y, z) volume, as data_to_sitk_image takes it"""
    return sitk.GetArrayFromImage(image).T


def ndl_patch_wise_prediction(
    device,
    model,
    data,
    patch_shape,
    iou_threshold,
    detections_per_patch,
    margin=0,
    batch_size=1,
    patient_name="",
):
    """
    Returns the spheres [x, y, z, radius, confidence] of a whole volume, in voxel coordinates, predicted by a model
    working on patches of patch_shape voxels. Patches overlap by 2 * margin voxels and only keep the spheres centered
    in their core (without margin); the volume is padded (with its minimum value) so that every voxel, borders
    included, lies in the core of exactly one patch.
    """
    torch.cuda.empty_cache()
    patch_shape = np.asarray(patch_shape, dtype=int)
    margin = np.broadcast_to(np.asarray(margin, dtype=int), patch_shape.shape)
    core = patch_shape - 2 * margin
    if np.any(core <= 0):
        raise ValueError(f"margin {margin.tolist()} leaves no core in patches of {patch_shape.tolist()} voxels")
    image_shape = np.asarray(data.shape)
    n_tiles = np.ceil(image_shape / core).astype(int)
    pad_after = n_tiles * core - image_shape + margin
    padded = np.pad(data, list(zip(margin, pad_after)), mode="constant", constant_values=data.min())
    starts = np.mgrid[: n_tiles[0], : n_tiles[1], : n_tiles[2]].reshape((3, -1)).T * core

    patient_spheres = []
    loop = tqdm(range(0, len(starts), batch_size), leave=True, unit="batch", total=math.ceil(len(starts) / batch_size))
    for j in loop:
        idx = starts[j : j + batch_size]
        batch = np.stack([padded[(np.newaxis,) + tuple(slice(s, s + p) for s, p in zip(i, patch_shape))] for i in idx])
        outputs = model(torch.from_numpy(batch).float().to(device))
        outputs = torch.cat((outputs[0][1], outputs[0][0]), -1).to("cpu")  # [x, y, z, radius, confidence]

        for patch_outputs, start in zip(outputs, idx):  # each patch of the batch
            spheres = cells_to_spheres(patch_outputs, obj_thresh=0.01, input_shape=list(patch_shape))
            spheres = [sphere for sphere in spheres if np.all((sphere[:3] >= margin) & (sphere[:3] < margin + core))]
            spheres = non_max_suppression(spheres, iou_threshold=iou_threshold)
            if detections_per_patch is not None:
                spheres = spheres[:detections_per_patch]

            # from patch to volume coordinates: the patch starts at start - margin in the unpadded volume
            offset = start - margin
            for sphere in spheres:
                sphere[0:3] = [float(c + o) for c, o in zip(sphere[0:3], offset)]
            patient_spheres += [sphere for sphere in spheres if np.all(np.asarray(sphere[:3]) < image_shape)]

        loop.set_description(f"{patient_name} ({len(starts)} patches; {len(patient_spheres)} detections)")
    return patient_spheres


def ndl_run_validation_case(
    pat_dict,
    device,
    size,
    dim,
    output_dir,
    patient_name,
    model,
    margin,
    batch_size,
    mirror_axes,
    iou_threshold,
    detections_per_patch,
    patient_wise_tta,
):
    """
    Runs a test case and writes predicted images to file.
    :param pat_dict: info about the patient data. Dictionary as returned by deepanedet.data.io.read_patient_data_base (including the 'dir' key)
    :param size: 3-tuple that provides the size in mm of a patch (patch shape will be extracted from model info)
    :param output_dir: Where to write prediction images.
    :param output_label_map: If True, will write out a single image with one or more labels. Otherwise outputs
    the (sigmoid) prediction values from the model.
    :param threshold: If output_label_map is set to True, this threshold defines the value above which is
    considered a positive result and will be assigned a label.
    :param model: model to use for prediction
    """
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir))

    affine = pat_dict["affine"]
    test_data = pat_dict["data"]

    # respace the input volume
    old_spacing = np.linalg.norm(affine, axis=0)[:3]
    new_spacing = np.array(size) / np.array(dim)
    vol = resample_to_spacing(test_data, old_spacing, new_spacing)

    spheres1 = ndl_patch_wise_prediction(
        device,
        model=model,
        data=vol,
        patch_shape=dim,
        margin=margin,
        batch_size=batch_size,
        patient_name=patient_name,
        iou_threshold=iou_threshold,
        detections_per_patch=detections_per_patch,
    )

    spheres1 = spheres_to_metric(spheres1, set_affine_spacing(affine, new_spacing))
    if not os.path.exists(os.path.join(output_dir, "F")):
        os.makedirs(os.path.join(output_dir, "F"))

    save_spheres(spheres1, os.path.join(output_dir, "F", patient_name + ".json"))

    # Test time Augmentation
    if patient_wise_tta and len(mirror_axes) > 0:
        for axis in mirror_axes:
            data = np.flip(vol, axis=axis)
            spheres2 = ndl_patch_wise_prediction(
                device,
                model=model,
                data=data,
                patch_shape=dim,
                margin=margin,
                batch_size=batch_size,
                patient_name=patient_name,
                iou_threshold=iou_threshold,
                detections_per_patch=detections_per_patch,
            )

            if not os.path.exists(os.path.join(output_dir, f"F{axis}")):
                os.makedirs(os.path.join(output_dir, f"F{axis}"))

            spheres2 = flip_coords(spheres=spheres2, axis=axis, dimension=data.shape)
            spheres2 = spheres_to_metric(spheres2, set_affine_spacing(affine, new_spacing))
            save_spheres(spheres2, os.path.join(os.path.join(output_dir, f"F{axis}"), patient_name + ".json"))
            # keep only intersected spheres
            spheres1 = keep_only_intersected_spheres(spheres1, spheres2, min_iou_threshold=0.1)
    save_spheres(spheres1, os.path.join(output_dir, patient_name + ".json"))


def ndl_run_validation_cases(
    pat_list,
    device,
    model,
    patch_size,
    patch_dim,
    input_volume,
    normalization,
    output_dir=".",
    margin=0,
    batch_size=1,
    mirror_axes=[0],
    do_tta=False,
    iou_threshold=0.01,
    detections_per_patch=None,
):
    """
    This function loops over a list of patient directories, predicts and saves the predictions for each patient.
    Parameters:
        - pat_list (list): a list of patient directories to be predicted
        - device (torch.device): the device (e.g., CPU or GPU) on which to run the model
        - model: the neural network model to use
        - patch_size (list): the size of the patches to extract from the input image
        - patch_dim (list): the patch dimension (e.g., 96x96x96)
        - input_volume (String): init volume
        - normalization (String): Normal or Linear
        - output_dir (String): the directory path in which to save the output files
        - margin (list): the size of the margin to add around each patch during processing (16x16x16)
        - batch_size (integer): the number of patches to process in each batch
        - do_tta (boolean): whether to perform test-time augmentation (TTA) during processing
        - mirror_axes (list): a list of axes to mirror the input image along if TTA is True
        - iou_threshold (float): the intersection over union (IoU) threshold above which two aneurysm detections are considered a match
        - detections_per_patch (integer): the maximum number of aneurysm detections to output for each patch
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for p in pat_list:
        patient_name = dio.patient_name(p)
        if not os.path.isfile(os.path.join(output_dir, patient_name) + ".json"):
            pat_db = dio.read_patient_data_base(
                [p], volume=input_volume, pts_aneurysm=None, normalize=normalization, log=False
            )
            ndl_run_validation_case(
                pat_db[0],
                device,
                output_dir=output_dir,
                patient_name=patient_name,
                model=model,
                size=patch_size,
                dim=patch_dim,
                margin=margin,
                batch_size=batch_size,
                mirror_axes=mirror_axes,
                patient_wise_tta=do_tta,
                iou_threshold=iou_threshold,
                detections_per_patch=detections_per_patch,
            )
