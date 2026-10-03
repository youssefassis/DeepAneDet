import numpy as np
import sklearn.neighbors as skn
import scipy.ndimage as sndi
import skimage.morphology as skim
import skimage.measure as skme
from scipy.ndimage import label
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

def extractPointsThres(vol,vox2met,thresLow, thresHigh):
    '''
    return points in vol whose value is between thresLow (strictly)
    and thresHigh (loosely)
    return points coordinates (p) and values at these points (v)
    p is returned as a Nx3 array (or D is len(vol.shape) == D)
    and v is returned as 1xN-array
    '''
    idx=np.nonzero(np.logical_and(vol>thresLow,vol<=thresHigh))
    v=vol[idx]
    p=vox2met[:3,:]@np.vstack((idx,np.ones(len(v))))+(0.5*vox2met[:3,:3]@np.ones(3)).reshape((-1,1))

    return p.T,v

def pointsInRadius(q,p,r):
    '''
    return the indices of all points from p within distance r from q
    (p=point set, q=query point)
    p should be a Nx3 array and q a 3-point or Nx3 array of points
    (work ok in dimensions D!=3 too)
    '''
    if len(q.shape) == 1:
        Qq=np.array(q)[np.newaxis,:]
    else:
        Qq=q
    tree=skn.KDTree(p)
    return tree.query_radius(Qq,r)[0]

def selectPoints(vol,vox2met,thresLow,r,thresHigh=None,fPoints=None,
                 nbPoints=None, extractType='Vessels'):
    '''
    return p: where p is a Nx3 array of the coordinates of points above threshold thres,
              such that any two pair of points are at least at a distance r
    If fPoints is not None, it should be a Nx3 array of forbidden points (coordinates).
    In that case, all points within a distance of each point in that list are
    removed.
    If nbPoints is provided (a number), then at most nbPoints are returned.
    The function starts with the brightest allowable point and review points with
    decreasing value. As a consequence, v should be sorted (decreasing) on output

    Example use:
    d=ni.load('volume.nii')
    vol=np.asarray(d.dataobj)
    vox2met=vol.affine
    T=np.percentile(vol,95) # threshold to get the 5% brigtest voxels
    fPoints=IO.readFcsv('markers.fcsv')
    p=selectPoints(vol,vox2met,T,20,fPoints.values(),100)
    -> this extracts the 100 brightest points, among the 5% brightest points in the
       volume stored in 'volume.nii' such that no two points are within a distance
       20 mm from each other and no point is within 20 mm from points read in file
       'markers.fcsv' 
    '''
    if thresHigh is None:
        thresHigh=np.max(vol.ravel())
    p,v=extractPointsThres(vol,vox2met,thresLow,thresHigh)

    # compute volume boundaries in metric space
    #E=vox2met[:3,:]@np.vstack((np.array([0,0,0,1]),np.array(vol.shape)))
    #cm=np.min(E,axis=0)
    #cM=np.max(E,axis=0)
    # remove points too close to boundaries
    if (nbPoints is None) or (nbPoints > len(v)):
        nbPoints = len(v)

    if extractType=='Vessels':
        order=np.argsort(v)[::-1] # pick points with decreasing voxel values
    else:
        order=np.arange(len(v))
        np.random.shuffle(order) # pick points randomly
    removed=np.zeros(len(v),dtype=np.bool)
    tree=skn.KDTree(p)

    if not fPoints is None:
        if len(fPoints.shape) == 1:
            fPoints = fPoints[np.newaxis,:]
        i=tree.query_radius(fPoints,r)[0]
        removed[i]=True

    ret=np.empty((0,3))

    n=0
    for idx in order:
        if not removed[idx]:
            q=p[idx,:].copy()
            i=tree.query_radius(q[np.newaxis,:],r)[0]
            removed[i]=True
            ret=np.vstack((ret,q[np.newaxis,:]))
            n=n+1
            if n == nbPoints:
                return ret
    return ret

def getBall(r):
    '''
    return a structure element shaped as a ball of radius r.
    can be used with skimage.morphology operators
    '''
    x,y,z=np.ogrid[-r:r+1,-r:r+1,-r:r+1]
    return ((x*x+y*y+z*z)<=r*r).astype(np.uint8)

def removeSkullMask(vol,percent=80):
    ''' 
    Skull Stripping operation
    '''
    # compute gradient map and threshold it to its 80th percentile
    edges=sndi.gaussian_gradient_magnitude(vol,sigma=3)
    idx=np.nonzero(edges>=np.percentile(edges,percent))
    # sorts the indices so that we can review lines in the volume
    # according to their y and z coordinates in the
    # correct order (y first in increasing order, and then z): 
    # x coordinates are naturally ordered so that, for a given (y,z) pair, 
    # the first encountered x value is the column index of the first non-null 
    # voxel along the (y,z) line, and the last encountered value is the last
    # non voxel along this line
    idx=sorted(np.vstack(idx).T,key=lambda x: (x[1], x[2]))
    # review lines according to (y,z) coordinates
    # get first and last non-null voxel and then, set all voxels between 
    # these two extremities to 1 in the mask 
    g=(i for i in idx)
    mask=np.zeros(vol.shape)
    try:
        n=1
        j0,j1,j2=next(g)
        while True:
            m=j0
            i0,i1,i2=j0,j1,j2
            while i1==j1 and i2==j2:
                M=j0
                j0,j1,j2=next(g)
                n+=1
            mask[m:M,i1,i2]=1
    except:
        pass

    # erode this mask to remove the skull
    selem=getBall(2)
    for _ in range(15):
        mask=skim.binary_erosion(mask,footprint=selem).astype(np.uint8)
    # return this mask
    return mask


def ConnectedComponents2Spheres(vol, vox2met, min_score=0.01):
    '''
    This function performs connected component labeling on a 3D volume and extracts spheres from the connected components.

    Parameters:
        vol: a 3D numpy array representing the volume.
        vox2met: a 4x4 numpy array representing the transformation matrix from voxel to metric coordinates.
        min_score: a float representing the minimum score threshold for voxel values. Voxel values less than this threshold are set to 0.
    
    The function first threshold the input volume by setting voxel values less than min_score to 0. 
    Then it performs connected component labeling on the thresholded volume using the label function
    from the scipy.ndimage module, which returns a labeled array where all connected regions are 
    assigned the same integer value. The function then iterates over each connected component and 
    performs the following steps:
        1- Creates a binary mask of the connected component by setting all voxel values not equal 
        to the current connected component to 0.
        2- Computes the confidence score for the connected component as the maximum value in the 
        binary mask.
        3- Extracts the positions of voxels in the connected component (in voxel coordinates) and
        transforms them to metric coordinates.
        4- Computes the center of gravity of the connected component in metric coordinates.
        5- Computes the radius of the sphere as half the difference between the maximum and minimum
        metric coordinates of the connected component.
        6- Appends the center coordinates, radius, and confidence score to the spheres array.

    The output of the function is the spheres array, and the points array. Each array
    is a numpy array with N rows, one per connected component.

    A row in the spheres array represents a sphere encoded with 5 values:
        - cog: a 3-element numpy array representing the center of gravity of the sphere in metric coordinates.
        - r: a float representing the radius of the sphere in metric units.
        - score: a float representing the confidence score of the sphere.
    
    A row in the points array has 6 components to store a pair of 3D points.
    These points are the farthest apart from each other in the connected component.
    They can only make sense in the case of a manual voxel-wise segmentation. 
    In the case of a sphere, the result is unpredictable, though both points always
    form a diameter of the sphere.
    '''
    
    # thresholding: remove voxel values less than min_score
    vol[vol < min_score] = 0
    # label CC in volume
    labels, ncc_pred = label(vol,structure=np.ones((3,3,3)))
    spheres = np.empty((0, 5))
    points = np.empty((0, 6))
    for i in range(ncc_pred):
        ## Confidence score
        score = vol[labels == i+1].max()
        
        # extract positions of voxels in CC (in voxel coords)
        vpos=np.vstack(np.where(labels == i+1))
        # transform in metric coords
        mpos = (vox2met@np.vstack((vpos, np.ones(vpos.shape[1]))))[:3,:]
        ## compute center of gravity
        cog = np.mean(mpos,axis=1)
        
        ## compute the radius: farthest distance between points
        # keep only points on the convex hull (beat the N^2 complexity)
        hull = ConvexHull(mpos.T)

        # Extract the points forming the hull
        hullpoints = mpos[:,hull.vertices]

        # Naive way of finding the best pair in O(H^2) time if H is number of points on
        # hull
        hdist = cdist(hullpoints.T, hullpoints.T, metric='euclidean')

        # Get the farthest apart points
        bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
        P1=hullpoints[:,bestpair[0]]
        P2=hullpoints[:,bestpair[1]]
        
        # compute radius
        r=0.5*np.linalg.norm(P1-P2)

        ## recompute points along the same direction as P2-P1 and going through cog
        # warning: this does not ensure P1 is in the neck and P2 on the dome
        v=(P2-P1)/2
        P1=cog-v
        P2=cog+v

        # alternative: through volume
        #        voxel_size = np.linalg.norm(vox2met[:3,:3],axis=0)
        #        voxel_volume = np.prod(voxel_size)
        #        sphere_volume = vpos.shape[1] * voxel_volume
        #        r = (sphere_volume * 3/(4*np.pi))**(1/3)
        spheres = np.vstack((spheres, np.hstack((cog, r, score))))
        points = np.vstack((points, np.hstack((P1,P2))))

    return spheres, points

