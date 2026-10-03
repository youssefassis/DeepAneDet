import os, math, json, sys
import torch

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import Data.IO as dio
import utils
from helpers import cells_to_spheres, non_max_suppression, spheres_to_metric, save_spheres, keep_only_intersected_spheres, flip_coords

def calculate_origin_offset(new_spacing, old_spacing):
    '''
    This is a Python function that calculates the origin offset between two sets
    of spacing values. The function takes in two arguments: new_spacing, which is
    a list of three spacing values representing the new spacing in x, y, and z 
    dimensions, and old_spacing, which is a list of three spacing values 
    representing the old spacing in x, y, and z dimensions.
    '''
    return np.subtract(new_spacing, old_spacing)/2

def set_affine_spacing(affine, spacing):
    '''
    This function adjusts the spacing of a given affine transformation matrix. 
    The function takes in two arguments: 
        - affine, which is a 4x4 NumPy array representing the affine transformation matrix, 
        - spacing, which is a list of three values representing the desired spacing in x, y, and z dimensions.
    '''
    def get_spacing_from_affine(affine):
        RZS = affine[:3, :3]
        return np.sqrt(np.sum(RZS * RZS, axis=0))
    scale = np.divide(spacing, get_spacing_from_affine(affine))
    affine_transform = np.diag(np.ones(4))
    np.fill_diagonal(affine_transform, list(scale) + [1])
    return np.matmul(affine, affine_transform)

def sitk_resample_to_spacing(image, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear, default_value=0.):
    '''
    This function takes a SimpleITK image, and resamples it to a new spacing. The 
    resampling is performed using an interpolation method specified by interpolator,
    and any pixels that are not part of the original image are assigned a value 
    specified by default_value.
    '''
    zoom_factor = np.divide(image.GetSpacing(), new_spacing)
    new_size = np.asarray(np.ceil(np.round(np.multiply(zoom_factor, image.GetSize()), decimals=5)), dtype=np.int16)
    offset = calculate_origin_offset(new_spacing, image.GetSpacing())
    reference_image = sitk_new_blank_image(size=new_size, spacing=new_spacing, direction=image.GetDirection(),
                                           origin=image.GetOrigin() + offset, default_value=default_value)
    return sitk_resample_to_image(image, reference_image, interpolator=interpolator, default_value=default_value)

def sitk_resample_to_image(image, reference_image, default_value=0., interpolator=sitk.sitkLinear, transform=None, output_pixel_type=None):
    '''
    This function resamples a SimpleITK image to match the size, spacing, direction,
    and origin of a reference image. It takes as input the image to be resampled, 
    the reference_image to which the image is resampled, the default_value for 
    out-of-bounds pixels, the interpolator used for resampling, the transform to be 
    applied to the image, and the output_pixel_type. It returns the resampled image 
    in the same format as the output_pixel_type.
    '''
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

def sitk_new_blank_image(size, spacing, direction, origin, default_value=0.):
    '''
    This function creates a new blank SimpleITK image with specified size, 
    spacing, direction, origin, and default value. The image data is initialized
    as a 3D numpy array with the given size, and all elements are set to the default
    value. The returned object is a SimpleITK image object, which can be used
    for various image processing tasks.
    '''
    image = sitk.GetImageFromArray(np.ones(size, dtype=np.float64).T * default_value)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image

def resample_to_spacing(data, spacing, target_spacing, interpolation="linear", default_value=0.):
    '''
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
    '''
    image = data_to_sitk_image(data, spacing=spacing)
    if interpolation == "linear":
        interpolator = sitk.sitkLinear
    elif interpolation == "nearest":
        interpolator = sitk.sitkNearestNeighbor
    else:
        raise ValueError("'interpolation' must be either 'linear' or 'nearest'. '{}' is not recognized".format(
            interpolation))
    resampled_image = sitk_resample_to_spacing(image, new_spacing=target_spacing, interpolator=interpolator,
                                               default_value=default_value)
    return sitk_image_to_data(resampled_image)

def data_to_sitk_image(data, spacing=(1., 1., 1.)):
    '''
    This function takes a 3D numpy array data and a tuple spacing as input arguments 
    and returns a SimpleITK image. The data array is converted to a SimpleITK image 
    using the sitk.GetImageFromArray() function, and the spacing tuple is used to set
    the voxel spacing of the image using the SetSpacing() method. 
    '''
    if len(data.shape) == 3:
        data = np.rot90(data, 1, axes=(0, 2))
    image = sitk.GetImageFromArray(data)
    image.SetSpacing(np.asarray(spacing, dtype=np.float64))
    return image

def sitk_image_to_data(image):
    '''
    This function takes a SimpleITK image object as input and returns its pixel data
    as a numpy array. If the image has 3 dimensions, it is rotated 90 degrees 
    counterclockwise along the first and third axes to orient it in a consistent way.
    '''
    data = sitk.GetArrayFromImage(image)
    if len(data.shape) == 3:
        data = np.rot90(data, -1, axes=(0, 2))
    return data

def ndl_patch_wise_prediction(device, model, data, patch_shape, iou_threshold, detections_per_patch, margin=0, batch_size=1, patient_name= "", S=None, anchors=None):
    torch.cuda.empty_cache()
    image_shape = np.array(data.shape)
    patient_boxes = []
    if isinstance(margin, int): margin = np.asarray([margin] * len(image_shape))

    delta = np.array(patch_shape)-2*margin
    n_idx = np.floor((image_shape-2*margin)/delta).astype(np.int64)
    overflow = image_shape - (n_idx*delta+2*margin)
    start = overflow//2
    indices = np.mgrid[:n_idx[0], :n_idx[1], :n_idx[2]].reshape((3, -1)).T
    indices = indices * delta + start
    
    ######################################################################################################################################################################
    scaled_anchors = torch.tensor(anchors).reshape(len(S), len(anchors), 1) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, len(anchors), 1) if anchors else None
    ######################################################################################################################################################################
    
    total_batches = int(math.ceil(indices.shape[0] / batch_size))
    loop = tqdm(range(0, indices.shape[0], batch_size), colour='GREEN', leave=True, unit='batch', total = total_batches)
    
    for j in loop:
        k = min(j + batch_size, len(indices))
        idx = indices[j : k]
        batch = np.asarray([data[np.newaxis, i[0]:i[0]+patch_shape[0], i[1]:i[1]+patch_shape[1], i[2]:i[2]+patch_shape[2]] for i in idx])
        
        spheres = model(torch.from_numpy(batch).float().to(device))
        spheres = torch.cat((spheres[0][1], spheres[0][0]), -1).to('cpu') # first scale only: [x, y, z, radius, confidence]
        anchor = scaled_anchors[0] if scaled_anchors is not None else None

        for patch_spheres, origin in zip(spheres, idx): # each patch of the batch
            boxes = cells_to_spheres(predictions=patch_spheres,
                                     obj_thresh = 0.01,
                                     anchors = anchor,
                                     input_shape = list(batch.shape[-3:]),
                                     exp_radius=True,
                                     is_pred=True,
                                     criteria=None) # or "boxes"

            # filter boxes/ patch borders: according to detection center
            boxes = [box for box in boxes if (box[0] >= margin[0] and box[1] >= margin[1] and box[2] >= margin[2] and  box[0] < margin[0] + delta[0] and box[1] < margin[1] + delta[1] and box[2] < margin[2] + delta[2])]

            boxes = non_max_suppression(boxes, iou_threshold=iou_threshold)
            if detections_per_patch is not None: boxes = boxes[:detections_per_patch]

            # transform boxes coords from patch to the global volume
            for box in boxes:
                box[0] = box[0] + origin[0] # center x
                box[1] = box[1] + origin[1] # center y
                box[2] = box[2] + origin[2] # center z
            patient_boxes += boxes

        loop.set_description(f"{patient_name} ({indices.shape[0]} patches; {len(patient_boxes)} detections)")
    return np.zeros(data.shape).astype('float32'), patient_boxes


def ndl_run_validation_case(pat_dict, device, size, dim, output_dir, patient_name, model, margin, 
                            batch_size, mirror_axes, iou_threshold, detections_per_patch, patient_wise_tta, S=None, anchors=None):
    """
    Runs a test case and writes predicted images to file.
    :param pat_dict: info about the patient data. Dictionary as returned by Data.IO.read_patient_dabase (including the 'dir' key)
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

    affine = pat_dict['affine']
    test_data = pat_dict['data']

    # respace the input volume
    old_spacing = np.linalg.norm(affine, axis=0)[:3]
    new_spacing = np.array(size) / np.array(dim)
    vol = resample_to_spacing(test_data, old_spacing, new_spacing)

    prediction, spheres1 = ndl_patch_wise_prediction(device, 
                                                    model=model, 
                                                    data=vol, 
                                                    patch_shape=dim, 
                                                    margin=margin, 
                                                    batch_size=batch_size, 
                                                    patient_name=patient_name, 
                                                    S=S, 
                                                    anchors=anchors,
                                                    iou_threshold = iou_threshold,
                                                    detections_per_patch=detections_per_patch,
                                                    )

    spheres1 = spheres_to_metric(spheres1, set_affine_spacing (affine, new_spacing))
    if not os.path.exists(os.path.join(output_dir, "F")):
        os.makedirs(os.path.join(output_dir, "F"))

    save_spheres(spheres1, os.path.join(output_dir, "F", patient_name+'.json'))
    
    # Test time Augmentation
    if patient_wise_tta and len(mirror_axes) > 0 :
        for axis in mirror_axes:
            data = np.flip(vol, axis=axis)
            prediction, spheres2 = ndl_patch_wise_prediction(  device, 
                                                               model=model, 
                                                               data=data, 
                                                               patch_shape=dim,
                                                               margin=margin, 
                                                               batch_size=batch_size,
                                                               patient_name=patient_name,
                                                               S=S,
                                                               anchors=anchors,
                                                               iou_threshold = iou_threshold,
                                                               detections_per_patch=detections_per_patch,
                                                               )

            if not os.path.exists(os.path.join(output_dir, f"F{axis}")):
                os.makedirs(os.path.join(output_dir, f"F{axis}"))

            spheres2 = flip_coords(spheres = spheres2, axis = axis, dimension = data.shape)
            spheres2 = spheres_to_metric (spheres2, set_affine_spacing(affine, new_spacing))
            save_spheres(spheres2, os.path.join(os.path.join(output_dir, f"F{axis}"),  patient_name+'.json'))
            # keep only intersected spheres            
            spheres1 = keep_only_intersected_spheres(spheres1, spheres2, min_iou_threshold=0.1)
    save_spheres(spheres1, os.path.join(output_dir, patient_name+'.json'))

def ndl_run_validation_cases(pat_list, device, model, patch_size, patch_dim, input_volume, normalization, output_dir='.', margin=0, batch_size=1, mirror_axes=[0], do_tta=False, scales=[12], anchors=None, iou_threshold=0.01, detections_per_patch=None ):
    '''
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
        - scales (list): a list of scales at which to process the input image (used for multi-scale testing)
        - iou_threshold (float): the intersection over union (IoU) threshold above which two aneurysm detections are considered a match
        - detections_per_patch (integer): the maximum number of aneurysm detections to output for each patch
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for p in pat_list:
        patient_name = os.path.basename(p)
        
        pat_db = dio.read_patient_data_base([p], volume=input_volume, pts_aneurysm=None, normalize=normalization, log=False)        
        if "CHUV" in pat_db[0]["dir"]:
            patient_name = pat_db[0]["dir"][pat_db[0]["dir"].find('sub'): ].replace("/", '').replace('-', '').replace('_', '')
        if not os.path.isfile(os.path.join(output_dir, patient_name)+".json"):
            ndl_run_validation_case(pat_db[0],
                                    device, 
                                    output_dir = output_dir,
                                    patient_name = patient_name,
                                    model = model,
                                    size = patch_size,
                                    dim = patch_dim,
                                    margin = margin,
                                    batch_size = batch_size, 
                                    mirror_axes = mirror_axes,
                                    patient_wise_tta = do_tta,
                                    S = scales,
                                    anchors = anchors,
                                    iou_threshold=iou_threshold,
                                    detections_per_patch=detections_per_patch,
                                   )
