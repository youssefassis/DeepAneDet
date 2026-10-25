import json
import torch
import numpy as np
from nibabel.affines import apply_affine


def to_tuple(value, length: int = 1):
    try:
        iter(value)
        value = tuple(value)
    except TypeError:
        value = length * (value,)
    return value


def to_array(value, length=1):
    """
    This function takes a value and an optional length argument and
    returns an array of the specified length. It does this by first
    converting the value to a tuple of the specified length using
    the to_tuple function.
    """
    return np.array(to_tuple(value, length=length))


def cells_to_spheres(predictions, obj_thresh, input_shape):
    """
    Returns the spheres [x, y, z, radius, confidence], in patch voxels, of the grid cells whose confidence is at least
    obj_thresh.
    Parameters:
        - predictions: raw network outputs of one patch, (grid, grid, grid, 5) for x, y, z (offset in the cell),
          radius (log, in cells) and confidence (logit)
        - obj_thresh: threshold value for the object confidence score.
        - input_shape: patch shape, in voxels
    """
    predictions = torch.as_tensor(predictions)
    centers = torch.sigmoid(predictions[..., :3]).numpy()
    radii = torch.exp(predictions[..., 3]).numpy()
    confidences = torch.sigmoid(predictions[..., 4]).numpy()
    cell_size = np.asarray(input_shape) / np.asarray(predictions.shape[:3])

    spheres = []
    for cell in zip(*np.nonzero(confidences >= obj_thresh)):
        x, y, z = (np.asarray(cell) + centers[cell]) * cell_size
        spheres.append([float(x), float(y), float(z), float(radii[cell] * cell_size[0]), float(confidences[cell])])
    return spheres


def non_max_suppression(spheres, iou_threshold):
    """
    Filters out overlapping detections of the same aneurysm: keeps the most confident sphere
    and drops the ones overlapping it by at least iou_threshold, then repeats with the others.
    spheres: list of [x, y, z, radius, confidence]
    """
    if not isinstance(spheres, list):
        raise TypeError(f"Expected a list of spheres, got {type(spheres).__name__}")
    remaining = sorted(spheres, key=lambda x: x[-1], reverse=True)  # sort by confidence score
    kept = []
    while remaining:
        chosen = remaining.pop(0)
        remaining = [
            sphere
            for sphere in remaining
            if intersection_over_union(np.array(chosen[:4]), np.array(sphere[:4])) < iou_threshold
        ]
        kept.append(chosen)
    return kept


def intersection_over_union(prediction, truth):
    """
    computes the intersection-over-union volume between two spheres
    """

    def get_intersection(point1, point2):
        pos1, pos2 = point1[:3], point2[:3]
        r1, r2 = point1[3], point2[3]
        d = sum((pos1 - pos2) ** 2) ** 0.5

        # check they overlap
        if d >= (r1 + r2):
            return 0.0

        # check if one entirely holds the other
        elif d == 0:
            return 4.0 / 3.0 * np.pi * min(r1, r2) ** 3
        elif r1 > (d + r2):  # 2 is entirely contained in one
            return 4.0 / 3.0 * np.pi * min(r1, r2) ** 3
        elif r2 > (d + r1):  # 1 is entirely contained in one
            return 4.0 / 3.0 * np.pi * min(r1, r2) ** 3
        else:
            vol = (
                np.pi
                * (r1 + r2 - d) ** 2
                * (d**2 + ((2 * d * r1 - 3 * r1**2) + (2 * d * r2 - 3 * r2**2)) + 6 * r1 * r2)
            ) / (12 * d)
        return vol

    inter = get_intersection(prediction, truth)
    union = ((4 / 3) * np.pi * (prediction[3] ** 3) + (4 / 3) * np.pi * (truth[3] ** 3)) - inter
    return inter / (union + 1e-10)


def flip_coords(spheres, axis, dimension):
    """
    This function flips the coordinates of a list of spheres along a given axis.
    It takes in three arguments:
        - spheres: a list of sphere coordinates,
        - axis: an integer indicating the axis to flip along,
        - dimension: a list of integers representing the size of each dimension
        of the space the spheres are in.
    """
    for sphere in spheres:
        sphere[axis] = abs(dimension[axis] - 1 - sphere[axis])
    return spheres


def save_json(data: tuple, path: str, indent: int = 4):
    """
    This function saves a tuple of data as a JSON file to a specified file path.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def save_spheres(spheres, out_file):
    """
    This is a Python function that saves a list of sphere coordinates as a JSON file
    to a specified file path.
    The function takes in two arguments: spheres, which is a list of sphere coordinates,
    and out_file, which is a string representing the file path where the JSON file will
    be saved. The sphere's ostium and dome, center points, radius, and confidence score
    are extracted from the given sphere coordinates and added/saved to the dictionary.
    """
    data = {
        id: {"center": [float(c) for c in sphere[:3]], "radius": float(sphere[3]), "confidence": float(sphere[-1])}
        for id, sphere in enumerate(spheres)
    }
    save_json(data=data, path=out_file)


def keep_only_intersected_spheres(spheres1, spheres2, min_iou_threshold):
    if len(spheres1) == 0 or len(spheres2) == 0:
        return []

    results = []
    for sphere1 in spheres1:
        for sphere2 in spheres2:
            iou = intersection_over_union(np.array(sphere1[:4]), np.array(sphere2[:4]))  # [x, y, z, radius, confidence]

            if iou >= min_iou_threshold:
                # take the shere with high confidence + avg (center & radius)
                if sphere1[-1] >= sphere2[-1]:
                    results.append(sphere1)
                else:
                    results.append(sphere2)
    return non_max_suppression(sorted(results, key=lambda x: x[-1], reverse=True), iou_threshold=min_iou_threshold)


def spheres_to_metric(spheres, v2m):
    """
    This is a Python function that transforms a list of sphere coordinates
    from voxel space to metric space using an affine transformation matrix.
    The function takes in two arguments: spheres, which is a list of sphere
    coordinates in voxel space, and v2m, which is a 4x4 affine transformation
    matrix that converts voxel coordinates to metric coordinates.
    """

    def apply_transformation(elements, trans):
        return apply_affine(trans, elements)

    v2m = v2m.squeeze()
    vox_size = np.linalg.norm(v2m[:3, :3], axis=0)

    for sphere in spheres:
        sphere[:3] = apply_transformation(np.array(sphere[:3]), v2m)  # center
        sphere[3] = float(sphere[3] * vox_size[0])  # radius
    return spheres
