import numpy as np
from skimage.filters import threshold_otsu

def drawSphere(vol, vox2met, c, r, val=0):
    """ 
    draw ('burn') a 3D sphere in the volume (vol). 
    The sphere is given by its center coordinates (c) and radius r in RAS coordinates. 
    vox2met is the 4x4 affine transform between voxel and RAS coordinates.
    val is the voxel value used for burnt voxels
    """
    met2vox=np.linalg.inv(vox2met)
    # first find a bounding box to extract a voxel patch
    c=np.asarray(c)
    pminus=met2vox[:3,:]@np.append(c-r,1)
    pplus=met2vox[:3,:]@np.append(c+r,1)
    pmin=np.minimum(pminus,pplus)
    pmax=np.maximum(pminus,pplus)
    pmin=np.round(pmin).astype(np.int32)
    pmax=np.round(pmax).astype(np.int32)

    # bounding box for the volume
    vmin=np.zeros(3).astype(np.int32)
    vmax=np.asarray(vol.shape).astype(np.int32)-1

    # compute intersection
    pmin = np.maximum(pmin,vmin)
    pmax = np.minimum(pmax,vmax)+1 # previous points where included in intervals
    
    # check for non empty intersection
    if np.all(np.greater(pmax,pmin)):
        # compute distance patch
        # list of RAS coordinates for the points in in the bounding box
        x,y,z=np.mgrid[pmin[0]:pmax[0],pmin[1]:pmax[1],pmin[2]:pmax[2]]
        p=vox2met[:3,:]@np.vstack((x.ravel(),y.ravel(),z.ravel(),np.ones(len(x.ravel()))))
        # compute distance field
        d = np.linalg.norm(p-c[:,np.newaxis],axis=0)
        # compute indices for points in sphere
        i = np.where(d<=r)[0]
        # burn voxels
        vol[pmin[0]:pmax[0],pmin[1]:pmax[1],pmin[2]:pmax[2]].flat[i] = val

def getTruth(vol, vox2met, spheres, refined=False):
    '''
    This function takes a volume (3D ndarray) and a list of spheres (ndarray with shape (n, 4))
    as input, where each row represents a sphere with its center coordinates (x, y, z) and radius.
    The function then burns the spheres into the volume, replacing the background values with 0 
    and assigning a unique index (starting at 1) to the voxels inside each sphere. If two spheres
    intersect, the highest label is kept for the intersecting voxels. The function returns a 
    labeled image (3D ndarray) where the value of each voxel corresponds to the index of the 
    sphere it belongs to. If the parameter "refined" is True, the function applies a refinement 
    step by thresholding the input volume using Otsu's method, setting the background values in 
    the labeled image to 0, and returning the refined labeled image.
    '''
    t = np.zeros(vol.shape).astype(np.uint8)
    for i,s in enumerate(spheres):
        drawSphere(t,vox2met,s[:3],s[3],i+1)
    
    if refined:
        vol[t != 1] = 0
        binary = vol > threshold_otsu(vol)
        t [binary == 0] = 0
    return t