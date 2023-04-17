import numpy as np
import nibabel as ni
import pandas as pd
import os, json, glob, random, time
from tqdm import tqdm
import sklearn

from Volume.Selection import selectPoints, ConnectedComponents2Spheres
from utils import get_logger
from itertools import repeat

from joblib import Parallel, delayed

logger = get_logger("Data Preparation")
# the chosen file format is python pickle dump of a dict. This might not be the wisest choice, but
# hdf5 format is not reentrant, so that you cannot leverage threads to train model (seg fault)
# csv might be nice, but I am not convinced that tables are appropriate to store our data
# sql could be a very nice option... to consider when I have the time

# the data
# for each patient, we have:
# - an MRA: stored as nii.gz file -> extract the data matrix and the affine transform (4x4 matrix)
# - a list of segments (pairs of points): one for each aneurysm, the first point is the entry point
#   and the second point is on the surface of the aneurysm so that the middle segment point is the
#   center of the aneurysm and the segment length gives the diameter of the approximating sphere
# - a list of points, outside the aneurysms, where patches should extracted (location of the patch centers)
# each point is provided in a metric space, and the affine transform metric to voxel coordinate transform

def fetch_patient_dirs(base_dir):
    '''
    list all directories in base_dir that meet name format 'P????' and contain a file named config.json
    primary use it to generate a list of patient directory names, with base_dir=config['base_dir'] (where
    config is read from a base config.json file, see PreprocessData.ipynb)
    '''
    return [os.path.dirname(subject_conf) for subject_conf in glob.glob(os.path.join(base_dir, 'P????', 'config.json'))]

def apply_transformation(elements, trans):
    '''
    Parameters:
    - "elements" are 3D point coordinates.
    - "trans" is a numpy array representing an affine transformation matrix.
    
    The function applies the affine transformation to the input "elements" using the apply_affine
    function from the nibabel library and returns the transformed array of elements.
    '''
    return ni.affines.apply_affine (trans, elements)

def read_nii_from_file(fname):
    '''
    Read a nii volume and return a 2-tuple with the voxel data (np array) and the voxel2metric tranform
    '''
    assert os.path.isfile(fname), f"nii file not found {fname}"
    nv = ni.load(fname)
    return nv.get_fdata(), nv.affine

def save_nii_to_file(fname, vol, vox2met):
    '''
    The transformation matrix from voxel coordinates to physical (metric) coordinates is passed 
    as "vox2met", which is also used to define the header information of the NIfTI file.
    Specifically, the function creates a NIfTI1Image object from the input "vol" and "vox2met" 
    using the nibabel library. Then, it saves the resulting image object to a file with the given
    filename "fname" using the save() function from the nibabel library.
    '''
    ni.save(ni.Nifti1Image(vol, vox2met), fname)
    
def read_points_from_csv(fname, nb = None):
    '''
    returns a list of points, read from a csv file. Coordinates are stored in columns named 'x', 'y' and 'z'
    '''
    assert os.path.isfile(fname), f"CSV file not found {fname}"
    try:
        try:
            d = pd.read_csv(fname, usecols = ['x','y','z', 'type'])
            if nb is not None:
                vessel = sklearn.utils.shuffle(d[d["type"]=='Vessel'])[:nb]
                parenchyma = sklearn.utils.shuffle(d[d["type"]=='Parenchyma'])[:nb]
                d = pd.concat([vessel, parenchyma])
            return d.drop(columns=['type']).to_numpy()
        except:
            d = pd.read_csv(fname, usecols = ['x','y','z'])
        return d.to_numpy()
    except:
        return None

def points_to_spheres(p):
    '''
    p: list of 3D points as a Nx3 array, where N is even, so that every two points are considered as segment
    returns: a list of N/2 x 4 values: the first 3 values are the center of each segment (i.e. center of the sphere),
             and the 4th value is its half-length (i.e. radius of the sphere)
    '''
    if p.shape[1] != 3:
        raise ValueError('Input array must be a list of 3D points')
    if p.shape[0]%2 != 0:
        raise ValueError('Input array must contain an even number of points')

    c = 0.5 * (p[::2] + p[1::2])
    a = p[::2] - p[1::2]
    r = np.sqrt(np.einsum('ij,ij->i', a, a))/2
    return np.hstack((c,r.reshape((-1, 1))))

def points_to_spheres_regression(p):
    '''
    p: list of 3D points as a Nx3 array, where N is even, so that every two points are considered as segment
    returns: a list of N/2 x 4 values: the first 3 values are the center of each segment (i.e. center of the sphere),
             and the 4th value is its half-length (i.e. radius of the sphere)
    '''
    if p.shape[1] != 3:
        raise ValueError('Input array must be a list of 3D points')
    if p.shape[0]%2 != 0:
        raise ValueError('Input array must contain an even number of points')

    c = 0.5 * (p[::2] + p[1::2])
    a = p[::2] - p[1::2]
    r = np.sqrt(np.einsum('ij,ij->i', a, a)) / 2
    return np.hstack((c, r.reshape((-1, 1)), p[::2], p[1::2]))

def read_patient_data(vol_file, ane_file, normalize=None):
    '''
    returns a dict with keys 'data','affine','aneurysms'
    'data': volume voxel data, as returned by read_nii_from_file(vol_file)
    'affine': voxel2metric affine transform, as returned by read_nii_from_file(vol_file)
    'aneurysms': 2Nx3 array of aneurysms defining points (one entry and one end point for each aneurysm, hence the 2N)
    'normalize': volume data normalization. Accepted values are:
        - None or 'None'->no normalization
        - 'Linear': linear rescaling between 0 and 1
        - 'Normal': rescaling to 0-mean and unit-variance data
    '''
    assert normalize in ['Linear', 'Normal', None]
    
    d, affine = read_nii_from_file(vol_file)
    d = d.astype(np.float64)
        
    s = None
    if ane_file is not None: # Patient with no aneurysms (F.csv)
        s = read_points_from_csv(ane_file)

    # should we use scikit.learn.normalize (or transform)?
    if normalize == 'Linear':
        d = normalize(d)
    elif normalize == 'Normal':
        d = standardize(d)

    return {'data':d, 'affine':affine, 'aneurysms':s}

def normalize(vol):
    '''
    Normalizes the data between [0, 1] of the input volume
    '''
    m = np.min(vol)
    M = np.max(vol)
    vol = (vol - m ) / (M - m)
    return vol

def standardize(vol):
    '''
    Z-score Normalization of the input volume
    '''
    m = np.mean(vol)
    std = np.std(vol)
    vol = (vol - m) / std
    return vol

def read_patient_data_from_config(patient_path, volume, pts_aneurysm, normalize):
    ''' same as read_patient_data, but vol_file and ane_file are read from a json config file under keys
    'init volume'
    '''
    cfg_name = os.path.join(patient_path, 'config.json')
    with open(cfg_name,'r') as f:
        c = json.load(f)
    return dict({'dir': patient_path}, **(read_patient_data(vol_file = os.path.join(patient_path, c[volume]),
                             ane_file = os.path.join(patient_path, pts_aneurysm) if (pts_aneurysm is not None and os.path.isfile(os.path.join(patient_path, pts_aneurysm))) else None ,
                             normalize = normalize
                             )))

def read_patient_data_base(pat_list, volume='init volume', pts_aneurysm='F.csv',normalize=None, label="Training", log=False, workers=1):
    if log:
        start_time = time.time()
        logger.info(f"{label} dataset: {len(pat_list)} patients (volume='{volume}'; '{normalize}' normalization; Truth ={pts_aneurysm}); workers={workers}")

    if workers > 1:
        results = Parallel(n_jobs=workers, backend='loky')(delayed(read_patient_data_from_config)(path,
                                                                                                   volume,
                                                                                                   pts_aneurysm,
                                                                                                   normalize,
                                                                                                   ) for path in pat_list)
    else:
        results = [read_patient_data_from_config(path, 
                                                 volume, 
                                                 pts_aneurysm, 
                                                 normalize, 
                                                 ) for path in pat_list]
    if log:
        logger.info(f"Data loading took {round(time.time() - start_time, 2)} secs")
    return results

def add_points_to_patient_data(da, pname, nb=None):
    '''
    takes a patient data_base stored in an array of dictionaries (da) and adds a new key 'points' to each dictionary whose value is a
    Nx3 np.array of 3D points, as read in a file named pname in each patient directory (stored in key 'dir').
    Points should be stored in a CSV file with columns 'x','y' and 'z' (see read_points_from_csv()).
    The input array of dictionary is updated and returned.
    See read_patient_data_base() for more info on the dictionary structure.
    '''
    for d in da:
        d.update({'points':read_points_from_csv(os.path.join(d['dir'], pname), nb)})
    return da

def saveSplit(cfg, train_list, valid_list, test_list):
    '''
    saves a list of patients (directories) split in train, validation and test lists into a json file (cfg), with keys 'training list', 
    'validation list' and 'testing list'
    If the file already exists, it is updated, eventually with new keys as above, else it is created with these three keys.
    '''
    try:
        with open(cfg,'r') as f:
            d = json.load(f)
    except:
        d = {}
    d['training list'] = train_list
    d['validation list'] = valid_list
    d['testing list'] = test_list
    with open(cfg, 'w') as f:
        json.dump(d, f, indent=4)

def readSplit(cfg):
    '''
    reads a split list of patients (directories), in a json file with keys 'training list', 'validation list' and 'testing list'
    see also saveSplit()
    '''
    with open(cfg, 'r') as f:
        d = json.load(f)
    logger.info(f"Data splits: Training {len(d['training list'])}, Validation: {len(d['validation list'])}, Testing: {len(d['testing list'])}")
    return d['training list'], d['validation list'], d['testing list']


def csv2fcsv(csv_pathname):
    filename, file_extension = os.path.splitext(csv_pathname)
    assert file_extension == '.csv', "Please choose a file '.csv' extension"
    fcsv_fileOut = filename + ".fcsv"
    pcsv = pd.read_csv(csv_pathname)
    label=[f'{t}_{i}' for i,t in pcsv[['Unnamed: 0','type']].to_numpy()]
    df = pd.concat([pcsv,pd.Series(label,name='id')], axis=1)
    df = df[['id','x','y','z']]
    head=f'# Markups fiducial file version = 4.10\n# CoordinateSystem = 0\n# columns = {",".join(df)}'
    with open(fcsv_fileOut, 'w') as f:
        f.write(head)
        for i,l in df.iterrows():
            f.write(','.join([f'{v}' for v in l])+'\n')

def fcsv2csv(fcsv_file):
    filename, file_extension = os.path.splitext(fcsv_file)
    assert file_extension == '.fcsv', "Please choose a file with '.fcsv' extension"

    with open(fcsv_file, 'r') as f:
        l=f.readlines()

    # coordinate system
    cs=l[1].strip().split('=')[1].strip()

    # column names
    cn=list(map(str.strip,l[2].split('=')[1].split(',')))
    # data
    data=[s.strip().split(',') for s in l[3:]]
    # check coordinate systems and column names
    # RAS: if cs='RAS' or cs='0'
    # LPS: if cs='LPS' or cs='1'
    # not checking for position column names (could be x,y,z or l,p,s or r,a,s
    if cs == 'LPS' or cs == '1':
        for d in data:
            d[1] = str(-float(d[1]))
            d[2] = str(-float(d[2]))
    # change position column names
    cn[1:4] = ['x','y','z']
    if len(cn) < len(data[0]): 
        cn = cn + [None, None]

    # create data frame
    d=pd.DataFrame(data=data, columns=cn)
    # save the result
    d.to_csv(filename + ".csv", index=False)

def extractPointsFromPatient(patient, r=20, nbPoints=100, outfile="points.csv", randomPoints=False):
    '''
    This function extracts points from a patient's imaging volume. The points can either be randomly
    selected from the parenchyma or selected from vessels based on a threshold of the image intensity.
    The function takes the patient's directory, the distance (r) between any two points, the number
    of points (nbPoints) to extract, the name of the output file (outfile), and a boolean variable 
    for randomly selecting points. 
    The function reads the patient data volume and point locations from a JSON file. It extracts the
    points from the vessel and parenchyma based on the specified method and exports the points as a 
    CSV file, which is then converted to an FCSV file.
    '''
    os.chdir(patient)
    print('Load volume from disk: '+patient)
    with open("config.json", 'r') as f:
        d = json.load(f)
    vol, vox2met = read_nii_from_file(d['noskull volume'])
    print(f'Extracting points')
    fPoints = read_points_from_csv(d['pts aneurysm'])
    fp = points_to_spheres(fPoints)[:,:-1] # drop centers
    q = None
    if randomPoints:
        minThreshold, maxThreshold = np.percentile(vol[vol>0], 0), np.percentile(vol[vol>0], 100)
        print(f'\nRandom Points ', end='')
        p=selectPoints(vol, vox2met, thresLow=minThreshold, thresHigh=maxThreshold, r=r,
                             fPoints=fp, nbPoints=nbPoints, extractType='Parenchyma')
        print(f'{len(p)} random points')
        title = 'Random'
    else:
        T= np.percentile(vol[vol>0], 95)
        minThreshold = np.percentile(vol[vol>0], 50)
        maxThreshold = np.percentile(vol[vol>0], 90)
        p = selectPoints(vol, vox2met, thresLow=T, r=r, fPoints=fp, nbPoints=nbPoints, extractType='Vessels')
        title = 'Vessel'
        print(f'{len(p)} vessel points')
        q=selectPoints(vol,vox2met,thresLow=minThreshold,thresHigh=maxThreshold,r=r,fPoints=np.vstack((fp,p)),
                                nbPoints=nbPoints,extractType='Parenchyma')
        print(f'{len(q)} Parenchyma points')

    # export to csv using pandas
    #print(f'Exporting to CSV')
    ps = pd.DataFrame(p, columns=list('xyz'))
    ps['type'] = pd.Categorical([title] * len(p))
    if q is not None:
        qs = pd.DataFrame(q, columns=list('xyz'))
        qs['type']=pd.Categorical(['Parenchyma']*len(q))
        points=ps.append(qs)
    else:
        points = ps
    points.to_csv(outfile)
    csv2fcsv(outfile)

def extractPoints(dataPath, r = 20, nbPoints=100, outfile="points.csv", randomPoints = False):
    patients = fetch_patient_dirs(dataPath)
    for patient in patients:
        extractPointsFromPatient(patient, r=r, nbPoints=nbPoints, outfile=outfile, randomPoints=randomPoints)

def countAnev(p):
    """
    This function takes a patient directory path p (String) as input, and returns 
    the number of aneurysms found in the patient directory. 
    Here's a more detailed description of how the function works:
    1- The function changes the current working directory to the patient directory p.
    2- The function reads the config.json file in the patient directory and stores 
    the content in the variable c.
    3- The function reads the aneurysm locations from the file path specified in the 
    c['pts aneurysm'] key. It assumes that the file is in CSV format and uses the 
    read_points_from_csv() function to read the points from the file. The resulting 
    array aloc contains the locations of the aneurysms.
    4- The function changes the current working directory back to the parent directory.
    5- Finally, the function returns the number of aneurysms by dividing the length of
    aloc by 2. This is because each row in aloc represents the coordinates of two points
    that define a line segment representing an aneurysm.
    """
    os.chdir(p)
    with open('config.json', 'r') as f:
        c = json.load(f)
    # read aneurysm locations
    aloc=read_points_from_csv(c['pts aneurysm'])
    os.chdir('..')
    return len(aloc)/2

def SizeAnev(p):
    '''
    This function calculates the size of aneurysms in a given directory path p.
    1- Creates an empty list sizes to store the aneurysm sizes.
    2- Changes the current working directory to the given path p.
    3- Reads the config.json file in the current directory using json.load() and stores the content in variable c.
    4- Checks if the pts aneurysm file exists, and if so, reads the points from the file using a function 
    read_points_from_csv() (which is not defined in the code you provided).
    5- Calculates the size of each aneurysm by computing the Euclidean distance between pairs of points in the list
    points, and appends each size to the sizes list.
    
    Returns the sizes list.
    '''
    sizes = []
    os.chdir(p)
    with open('config.json', 'r') as f:
        c = json.load(f)
    # read aneurysm locations
    if c['pts aneurysm'] and os.path.isfile(c['pts aneurysm']):
        points = read_points_from_csv(c['pts aneurysm'])
        for i in range(0, points.shape[0], 2):
            p1, p2 = points[i], points[i+1]
            size = np.linalg.norm(p1 - p2, axis=0)
            sizes.append(size)
    return sizes


def get_patient_dir_from_name(patient_name):
    '''
    This function takes a patient name as input and returns the path to the patient's directory. 
    Here's a description of how the function works:
    1- If the patient name contains "P0", it is assumed to be from the CHRU or ADAM dataset. 
    The function first checks if a directory exists in the CHRU dataset with the patient name. 
    If it does, it returns the path to that directory. Otherwise, it checks for the patient name
    in various subdirectories of the ADAM dataset, including "Unique", "Basic", "Follow-up", and
    "healthy_patients". If it finds the patient name in one of these subdirectories, it returns
    the path to the corresponding directory.
    2- If the patient name does not contain "P0", it is assumed to be from the CHUV Lausanne dataset.
    The function constructs the path to the patient's directory by replacing the "sub" prefix with 
    "sub-" and taking the first seven characters of the patient name. It checks if the directory 
    exists in the "healthy_patients", "Sphere_annotation", or "Voxel_annotation" subdirectories of 
    the CHUV Lausanne dataset, in that order. If it finds the patient directory in one of these 
    subdirectories, it selects the first session directory in the patient directory (which is assumed
    to exist) and returns its path.
    '''
    # CHRU or ADAM
    if "P0" in patient_name:
        patient = os.path.join("/home/yassis/NeuroDL/Data/CHRU/", patient_name)
        if os.path.isdir(patient): # CHRU
            return patient
        else: # ADAM
            main_dir = "/home/yassis/NeuroDL/Data/ADAM/"
            if os.path.isdir(os.path.join(main_dir, "Unique", patient_name)): # unique folder
                return os.path.join(main_dir, "Unique", patient_name)
            elif os.path.isdir(os.path.join(main_dir, "Basic", patient_name)): # Basic folder
                return os.path.join(main_dir, "Basic", patient_name)
            elif os.path.isdir(os.path.join(main_dir, "Follow-up", patient_name)): # Follow-up folder
                return os.path.join(main_dir, "healthy_patients", patient_name)
            elif os.path.isdir(os.path.join(main_dir, "healthy_patients", patient_name)):
                return os.path.join(main_dir, "healthy_patients", patient_name)
    # CHUV Lausanne
    else:
        main_dir = "/home/yassis/NeuroDL/Data/CHUV/"
        patient = os.path.join(main_dir, "healthy_patients", patient_name.replace("sub", "sub-")[:7])
        if not os.path.isdir(patient):
            patient = os.path.join(main_dir, "Sphere_annotation", patient_name.replace("sub", "sub-")[:7])
            if not os.path.isdir(patient):
                patient = os.path.join(main_dir, "Voxel_annotation", patient_name.replace("sub", "sub-")[:7])
        # acces to session dir
        patient = [os.path.join(patient, f) for f in sorted(os.listdir(patient)) if "." not in f][0]
        return patient
