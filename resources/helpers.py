import os, json, sys, torch, math
import numpy as np    
from nibabel.affines import apply_affine
import Data.IO as dio


def to_tuple( value, length: int = 1):
    try:
        iter(value)
        value = tuple(value)
    except TypeError:
        value = length * (value,)
    return value
    
def to_array(value, length=1):
    '''
    This function takes a value and an optional length argument and 
    returns an array of the specified length. It does this by first
    converting the value to a tuple of the specified length using 
    the to_tuple function.
    '''
    return np.array(to_tuple(value, length=length))

def cells_to_spheres(predictions, obj_thresh, input_shape, anchors, is_pred, exp_radius, criteria):
    '''
    This function transforms the predicted spheres from the scale of the grid 
    (e.g. 12x12x12) to the patch scale (e.g. 96x96x96).
    Parameters:
        - Predictions: as a np.array
        - obj_thresh: threshold value for the object confidence score.
        - input_shape: as a list        
    '''
    assert criteria in [None, "boxes"], "Ooops criteria please"
    spheres = []
    nb_anchors = 1 if anchors is None else len(anchors)
    predictions = predictions if type(predictions) is np.ndarray else predictions.numpy()
    grid_h, grid_w, grid_d = predictions.shape[0:3] if nb_anchors == 1 else predictions.shape[1:4]

    if is_pred:
        predictions[..., -1:] = torch.sigmoid(torch.from_numpy(predictions[..., -1:])).numpy() # normalize confidence scores [0,1]

        if criteria is None: # Bounding Spheres (x, y, z, radius, confidence)
            predictions[..., :3] = torch.sigmoid(torch.from_numpy(predictions[..., :3])).numpy() # center (x, y, z) coords normalization
            if exp_radius:
                if anchors is None:
                    predictions[..., 3:4] = torch.exp(torch.from_numpy(predictions[..., 3:4])).numpy() # radius 's sphere                                
                else:
                    anchors = anchors.reshape(len(anchors), 1, 1, 1, 1)
                    predictions[..., 3:4] = (torch.exp(torch.from_numpy(predictions[..., 3:4])) * anchors).numpy() # radius
        else: # Bounding boxes : x, y, z, width, height, depth, confidence
            predictions[..., :3] = torch.sigmoid(torch.from_numpy(predictions[..., :3])).numpy() # center (x, y, z) coords normalisation
            if exp_radius: predictions[..., 3:6] = torch.exp(torch.from_numpy(predictions[..., 3:6])).numpy() # width, height, depth 's box

    for i in range(grid_h):
        for j in range(grid_w):
            for k in range(grid_d):
                row, col, dep = i, j, k
                for b in range(nb_anchors):
                    # confidence score
                    confidence = float(predictions[b, i, j, k, -1]) if anchors is not None else float(predictions[i, j, k, -1])
                    if confidence >= obj_thresh:
                        # Bounding Spheres (x, y, z, radius, confidence)
                        if criteria is None:
                            x, y, z, radius = predictions[i, j, k, :4] if anchors is None else predictions[b, i, j, k, :-1]
                            # center coords
                            x = (row + x) * (input_shape[0] / grid_h)
                            y = (col + y) * (input_shape[1] / grid_w)
                            z = (dep + z) * (input_shape[2] / grid_d)
                            radius = radius * (input_shape[0] / grid_h)
                            spheres.append([x, y, z, radius, confidence])

                        else: # Bounding Boxes (x, y, z, w, h, d, confidence)
                            # center coords
                            x, y, z, w, h, d = predictions[i, j, k, :-1]
                            x = (row + x) * (input_shape[0] / grid_h)
                            y = (col + y) * (input_shape[1] / grid_w)
                            z = (dep + z) * (input_shape[2] / grid_d)
                            w = w * (input_shape[0] / grid_w)
                            h = h * (input_shape[0] / grid_h)
                            d = d * (input_shape[0] / grid_d)
                            
                            #spheres.append([x, y, z, w, h, d, confidence])
                            # or
                            radius = np.max([w, h, d])
                            spheres.append([x, y, z, radius, confidence])
    return spheres

def non_max_suppression(bboxes, iou_threshold):
    '''
    This function performs non-maximum suppression (NMS) on a list of bounding spheres.
    NMS is a technique used in object detection to filter out overlapping detections of 
    the same object. The function takes in two arguments: bboxes, which is a list of 
    predictions, and iou_threshold, which is the threshold for the intersection over 
    union (IoU) metric.
    '''
    assert type(bboxes) == list
    if len(bboxes) == 0: return bboxes
    bboxes = sorted(bboxes, key=lambda x: x[-1], reverse=True)    # sort by confidence score
    bboxes_after_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        results = []
        for box in bboxes:
            if len(box) == 5: # bounding spheres
                center1, radius1 = box[0:3], box[3]
                center2, radius2 = chosen_box[0:3], chosen_box[3]
            else: # bounding boxes : [center (x, y, z), width, height, depth, confidence]
                center1, center2 = box[0:3], chosen_box[0:3]
                radius1 = max([box[3], box[4], box[5]])
                radius2 = max([chosen_box[3], chosen_box[4], chosen_box[5]])
            if intersection_over_union( np.array(center2+[radius2]), np.array(center1+[radius1])) < iou_threshold:
                results.append(box)
        bboxes = results[:]
        bboxes_after_nms.append(chosen_box)
    return sorted(bboxes_after_nms, key=lambda x: x[-1], reverse=True)

def intersection_over_union(prediction, truth):
    '''
    computes the intersection-over-union volume between two spheres
    '''
    def get_intersection(point1, point2):
        pos1, pos2 = point1[:3], point2[:3]
        r1, r2 = point1[3], point2[3]
        d = sum((pos1 - pos2) ** 2) ** 0.5
        
        # check they overlap
        if d >= (r1 + r2):
            return 0.
        
        # check if one entirely holds the other
        elif d == 0:
            return 4. / 3. * np.pi * min(r1, r2) ** 3
        elif r1 > (d + r2):  # 2 is entirely contained in one
            return 4. / 3. * np.pi * min(r1, r2) ** 3
        elif r2 > (d + r1):  # 1 is entirely contained in one
            return 4. / 3. * np.pi * min(r1, r2) ** 3
        else:
            vol = (np.pi * (r1 + r2 - d) ** 2 * (d**2 + ((2*d*r1 - 3*r1**2) + (2*d*r2-3 * r2**2)) + 6*r1*r2)) / (12*d)
        return vol
    inter = get_intersection(prediction, truth)
    union = ((4/3) * np.pi * (prediction[3]**3) +  (4/3) * np.pi * (truth[3]**3)) -  inter
    return inter / (union + 1e-10)

def flip_coords (spheres, axis, dimension):
    '''
    This function flips the coordinates of a list of spheres along a given axis.
    It takes in three arguments: 
        - spheres: a list of sphere coordinates,
        - axis: an integer indicating the axis to flip along,
        - dimension: a list of integers representing the size of each dimension 
        of the space the spheres are in.
    '''
    for sphere in spheres:
        sphere[axis] = abs (dimension[axis] - 1 - sphere[axis])
        if len(sphere)>5:
            sphere[axis+3] = abs (dimension[axis] - 1 - sphere[axis+3])
    return spheres

def save_json(data: tuple, path: str, indent: int = 4):
    '''
    This function saves a tuple of data as a JSON file to a specified file path. 
    '''
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)

def save_spheres(spheres, out_file):    
    '''
    This is a Python function that saves a list of sphere coordinates as a JSON file 
    to a specified file path. 
    The function takes in two arguments: spheres, which is a list of sphere coordinates,
    and out_file, which is a string representing the file path where the JSON file will
    be saved. The sphere's ostium and dome, center points, radius, and confidence score
    are extracted from the given sphere coordinates and added/saved to the dictionary.
    '''
    data = {}
    if len(spheres)==0:
        save_json(data = data, path = out_file)
        
    for id, cc in enumerate (spheres):
        if len(cc) == 5:
            data[id] = {"center" : list(cc[:3]), "radius" : float(cc[3]), "confidence": float(cc[-1])}
        else: # Bounding boxes 
            data[id] = {"center" : list(cc[:3]), "width" : float(cc[3]), "height" : float(cc[4]), "depth" : float(cc[5]), "confidence": float(cc[-1])}
        save_json(data = data, path = out_file)    
    
def keep_only_intersected_spheres(spheres1, spheres2, min_iou_threshold):
    assert type(spheres1) == type(spheres2) == list
    if len(spheres1) == 0 or len(spheres2) == 0: return []

    results = []
    for sphere1 in spheres1:
        for sphere2 in spheres2:
            if len(sphere2) == 5: #[x, y, z, radius, confidence]
                iou = intersection_over_union( np.array(sphere1[:4]), np.array(sphere2[:4]))
            else:
                # A revoir pour les bounding boxes !!!
                radius1 = np.linalg.norm(np.array(sphere1[0:3]) - (np.array(sphere1[0:3])+np.array(sphere1[3:6])))
                radius2 = np.linalg.norm(np.array(sphere2[0:3]) - (np.array(sphere2[0:3])+np.array(sphere2[3:6])))
                iou = intersection_over_union( np.array(sphere1[:3]+[radius1]), np.array(sphere2[:3]+[radius2]))
                
            if iou >= min_iou_threshold:
                # take the shere with high confidence + avg (center & radius)
                if sphere1[-1] >= sphere2[-1]:
                    results.append(sphere1)
                else : 
                    results.append(sphere2)
    return non_max_suppression(sorted(results, key=lambda x: x[-1], reverse=True), iou_threshold=min_iou_threshold)

def spheres_to_metric (spheres, v2m):
    '''
    This is a Python function that transforms a list of sphere coordinates 
    from voxel space to metric space using an affine transformation matrix. 
    The function takes in two arguments: spheres, which is a list of sphere 
    coordinates in voxel space, and v2m, which is a 4x4 affine transformation 
    matrix that converts voxel coordinates to metric coordinates.
    '''
    def apply_transformation(elements, trans):
        return apply_affine (trans, elements)
    assert isinstance(spheres, list)
    v2m = v2m.squeeze()    
    vox_size = np.linalg.norm(v2m[:3,:3], axis=0)
    
    for sphere in spheres:
        sphere[:3] = apply_transformation(np.array(sphere[:3]), v2m) # center
        if len(sphere) == 5: # x, y, z, radius, confidence
            sphere[3] = float(sphere[3] * vox_size[0]) # radius
        else: # Bounding boxes
            sphere[3] = float(sphere[3] * vox_size[0]) # width
            sphere[4] = float(sphere[4] * vox_size[1]) # height
            sphere[5] = float(sphere[5] * vox_size[2]) # depth
    return spheres
