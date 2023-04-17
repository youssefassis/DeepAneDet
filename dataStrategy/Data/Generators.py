import numpy as np
import random, json
import scipy.spatial.transform.rotation as sstr

import Volume.Patch as vp
import Volume.Edition as ved
import Data.IO as dio

from helpers import to_array

from utils import get_logger, get_number_of_steps 

logger = get_logger("Generators")

def generateTransforms(trans=None, scal=None, rot=None, center=None, disp=None):
    '''
    generate a augmentation transform as a pair of affine,disp (see Patch.getPath function)
    trans: translation amplitude. Can be a 3-vector (along x,y,z) or a scalar (same amplitude in all directions)
    rot: rotation amplitude (in degrees). Can be a 3-vector (around x,y,z) or a scalar (same angle amplitude around all axes)
    center: if set, center of rotation. Else, center is (0,0,0)
    disp: amplitude for the non-rigid (spline) deformation. Can be a 3-vector (along x,y,z) or a scalar (same amplitude in all directions). 
    Return affine, disp as a pair of transform parameters that can be used as (respectively) affine and disp arguments when calling Patch.getPatch
    If both trans and rot are None, then the returned affine is None
    If disp is None, then the returned disp is None
    (in compliance with Patch.getPatch default params)
    returned disp is a 3-tuple of displacements to apply on control points in the x,y and z dimensions. The shape of each tuple is (3,3,3) so that the deformation
    is controlled by a mesh of 3x3x3 control points superimposed on the patch (corners correspond). The returned disp makes sure that no displacement is 
    applied on the center point of the patch (disp[:][1,1,1]=0)
    ''' 
    if trans is None and rot is None and scal is None:
        affine = None
    else:
        t = np.zeros(3)
        if not trans is None:
            if np.isscalar(trans):
                t = to_array(value=trans, length=3)
            elif len(trans) == 3:
                t = trans
            else:
                raise TypeError("trans argument should be either a scalar or a 3-vector")
        t = t*(2*(np.random.random_sample(3)-0.5))

        s = np.ones((3, 3))
        if scal is not None:
            if isinstance(scal, list):
                assert scal[1] > 1, "max scaling factor should be greather than 1"
                #assert scal[0] < 1, "min scaling factor should be lower than 1"
                s = to_array(value = np.random.uniform(low = scal[0], high = scal[1], size = None), length = 3) # 0.6 -> 0.8 before
                s = np.diag(s)
                s[s == 0] = 1
            else:
                raise TypeError("scaling argument should be eithter a list")    
        
        r=np.zeros(3)
        if not rot is None:
            if np.isscalar(rot):
                r = to_array(value=rot, length=3)
            elif len(rot) == 3:
                r=rot
            else:
                raise TypeError("rot argument should either a scalar or a 3-vector")
        r = r*(2*(np.random.random_sample(3)-0.5))
        affine = np.eye(4)
        affine[:3,:3] = np.multiply(s, sstr.Rotation.from_euler('zyx',r,degrees=True).as_matrix())
        affine[:3,3] = t if center is None else t+center.ravel()@(np.eye(3)-affine[:3,:3]).T
        
    if not disp is None:
        if np.isscalar(disp):
            amp = to_array(value=disp, length=3)
        elif len(disp) == 3:
            amp=disp
        else:
            raise TypeError("disp argument should be either a scalar or a 3-vector")
        idx = amp[0]*2*(np.random.random_sample((3,3,3))-0.5)
        idy = amp[1]*2*(np.random.random_sample((3,3,3))-0.5)
        idz = amp[2]*2*(np.random.random_sample((3,3,3))-0.5)
        idx[1,1,1] = idy[1,1,1] = idz[1,1,1] = 0

        dx=np.zeros((5,5,5), dtype=np.float64)
        dy=np.zeros((5,5,5), dtype=np.float64)
        dz=np.zeros((5,5,5), dtype=np.float64)

        dx[1:-1,1:-1,1:-1]=idx
        dy[1:-1,1:-1,1:-1]=idy
        dz[1:-1,1:-1,1:-1]=idz
        disp = (dx, dy, dz)
    return affine, disp

def splitPatList(pat_list, training_pct=0.7, validation_pct=0.2, testing_pct=0.1):
    '''
    Split a list of patient directories into training, validation, and testing subsets according to given percentages.
    Parameters:
    pat_list (list of str): List of patient directory paths.
    training_pct (float): Percentage of patients to use for training (default 0.7).
    validation_pct (float): Percentage of patients to use for validation (default 0.2).
    testing_pct (float): Percentage of patients to use for testing (default 0.1).
    Returns:
    Tuple of three lists: training, validation, and testing patient directory paths.
    This function first computes the number of patients for each subset based on the input percentages,
    and then randomly shuffles the input list and splits it accordingly. The sum of the input percentages
    is normalized to 1 if necessary. The function returns three lists of patient directory paths for 
    training, validation, and testing, respectively.
    '''
    train_nb=round(len(pat_list) * training_pct/(training_pct+validation_pct+testing_pct))
    valid_nb=round((len(pat_list)-train_nb) * validation_pct/(validation_pct+testing_pct))
    random.shuffle(pat_list)
    return pat_list[:train_nb],pat_list[train_nb:train_nb+valid_nb],pat_list[train_nb+valid_nb:]


def generate_masks_nii(main_dir, pts_file="F.csv"):
    '''
    Generate Ground truth masks for patients from 3D images and point annotations.
    Parameters:
    main_dir (str): Path to the main directory containing patient directories.
    Returns:
    None.
    This function reads the initial volume image file and point annotations for each patient directory,
    converts the annotations contained in the F.csv files to spheres, and generates ground truth masks
    by filling the spheres with a label=1. The resulting masks are saved in NIfTI file format to the 
    patient directory as "mask_spheres.nii.gz".
    '''
    dir_pats = dio.fetch_patient_dirs(main_dir)
    for patient in dir_pats:
        with open(os.path.join(patient, "config.json"), 'r') as f:
            cfg = json.load(f)
            volume_name = os.path.join(patient, cfg["init volume"])

        vol, v2m = dio.read_nii_from_file(volume_name)
        spheres = dio.points_to_spheres(dio.read_points_from_csv(os.path.join(patient, pts_file)))
        truth = ved.getTruth(vol=vol, vox2met=v2m, spheres=spheres)
        dio.save_nii_to_file(os.path.join(patient, "mask_spheres.nii.gz"), truth, v2m)

def getPatches(pat_db, dup_ane, batch_size):
    '''
    Parameters:
    pat_db: patient data base (with points, as returned by IO.add_points_to_patient_data())
    dup_ane: number of duplications for each aneurysm center in the final point list
    batch_size: batch size used to generate the number of iterations
    Returns:
    list of samples, indexed by center points
    
    The function generates a new list of dictionaries to store the data on a point basis. 
    Each dictionary in the list contains the following keys: 'point', 'data', 'vox2met', 
    'aneurysms', 'affine', and 'disp'. The 'point' key represents the center for the patch, 
    the 'data' key represents the volume data, the 'vox2met' key represents the voxel to metric 
    transform, the 'aneurysms' key represents the approximating for the aneurysms in this volume, 
    and the 'affine' and 'disp' keys represent the parameters for data augmentation.
    '''

    # first generate the dictionary with the points stored for each patient
    p_list = [{'dir': d['dir'], 'point': p,'data': d['data'],'vox2met':d['affine'],'aneurysms':d['aneurysms'],'status':False} for d in pat_db for p in d['points']]
    
    # next generate a similar list with each point in 'aneurysms'
    a_list = [{'dir': d['dir'], 'point': p,'data': d['data'],'vox2met':d['affine'],'aneurysms':d['aneurysms'],'status':True} for d in pat_db for p in dio.points_to_spheres(d['aneurysms'])[:,:3]]
    
    # merge the two lists with duplicates for aneurysm points
    g_list = p_list + a_list*dup_ane

    return g_list, get_number_of_steps(len(g_list), batch_size)