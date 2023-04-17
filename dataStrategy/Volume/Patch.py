import sys
from helpers import to_array

import numpy as np
import scipy.ndimage as sndi
import scipy.optimize as sopt
import Volume.Edition as ved
import Data.IO as dio
from nibabel.affines import apply_affine
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes

def invTransPoint(p, center, size, affine=None, disp=None, order=3):
    '''
    This function takes as input a points p (of shape (3,-1)), a center,
    and size parameters defining a cube, and optional affine and deformation
    transformations. It applies the inverse of the transform defined by 
    first applying a non rigid deformation (grid deformation defined by disp)
    followed by an affine deformation (given by 4x4 matrix affine).
    The function returns the transformed point x.
    '''
    x = p.copy()
    # apply affine transform
    if affine is not None:
        D = np.linalg.inv(affine)
        x = (D[:3,:]@np.vstack((x,np.ones(x.shape[1]))))

    # then apply inverse of deformation
    if disp is not None:
        center = np.asarray(center)
        size = np.asarray(size)
        # determine abscissa relative to the cube
        minCorner=(center-size/2).reshape((3,1))
        res=((np.asarray(disp[0].shape)-1)/size).reshape((3,1))
        c=(x-minCorner) * res


        # compute non-rigid displacement
        x[0] -= sndi.map_coordinates(disp[0], c, order=order, mode='constant', prefilter=False).ravel()
        x[1] -= sndi.map_coordinates(disp[1], c, order=order, mode='constant', prefilter=False).ravel()
        x[2] -= sndi.map_coordinates(disp[2], c, order=order, mode='constant', prefilter=False).ravel()
    return x

def _euclideanDist(x,*args):
    '''
    This function calculates the Euclidean distance between two points in 3D 
    space, where the second point is transformed using an affine transformation
    followed by a non-rigid deformation (given by a displacement field) before
    calculating the distance. The function takes a vector x representing the 
    first point, and a list of arguments args containing the second point p, 
    the center and size of a cube center and size, an affine transformation 
    affine, and a displacement field disp.
    The function first uses the invTransPoint function to transform the input 
    points x using the inverse affine transformation and displacement field.
    It then calculates the Euclidean distance between the transformed point 
    q and the second point p, and returns the result.
    '''
    p = args[0]
    center = args[1]
    size = args[2]
    affine = args[3]
    disp = args[4]
    q = invTransPoint(x.reshape((3,-1)),center,size,affine,disp)
    d = p - q.ravel()
    res = np.sum(d*d)
    return res

def transPoint(p,center,size,affine=None,disp=None):
    '''
    apply the composite transform (see Patch.invTransPoint) to a set of points
    p stored in an 3xN array
    '''
    residual = 0 
    x = p.copy()
    if disp is not None:
        ores = sopt.minimize(_euclideanDist, x.ravel(), method='CG', options={'disp':False, 'maxiter':3}, args=(p.ravel(),center,size,affine,disp))
        residual = ores.fun        
        x = ores.x.reshape((3, -1))
    elif affine is not None:
        x=affine[:3,:]@np.vstack((x,np.ones(x.shape[1])))
    return x.T, residual

def getPatchAndAneurysms(vol, vox2met, center, size, dim, aneurysms, affine=None, disp=None):
    '''
    This function takes a 3D volume (vol) and extracts a patch, of a dimension dim (in voxels),
    around a given center with a given size (in mm) and vox2met transform. 
    It also applies an optional affine and non-rigid deformation disp. It then transforms any 
    aneurysms points located in the patch volume to the same coordinate space and computes their 
    centers and radii using dio.points_to_spheres(a). The coordinates of aneurysms that are 
    entirely inside the patch volume and mainted a distance of 3 voxels from the patch borders 
    are stored in kept_points. If any aneurysm is not entirely inside the patch volume, the 
    function returns fully_covered as False. The function also computes the residual loss 
    from the _euclideanDist function. 
    Finally, the function returns the patch volume, the corresponding vox2met 
    transform for the patch, the kept aneurysms coordinates in the patch kept_points,
    whether all aneurysms are fully inside the patch fully_covered, and the residual
    loss if applicable.
    '''
    v, v2m = getPatch(vol, vox2met, center, size, dim, affine, disp)

    # generate corresponding aneurysm point as truth
    kept_points = np.empty((0, 3), int)
    ignore = False
    
    loss = 0
    if aneurysms is not None:
        #   apply transform to aneurysms points. Need for transpose since points are stored in (-1,3) array
        a, loss = transPoint(aneurysms.T, center, size, affine=affine, disp=disp)
        #   compute centers and radii
        s = dio.points_to_spheres(a)
        
        # transform aneurysms coords to voxel space excluding aneurysms located outside the patch volume
        points_vox = (np.linalg.inv(v2m)[:3,:]@np.vstack((a.T, np.ones(a.shape[0])))).T
        points_vox = np.round(points_vox).astype(np.int32) # ? should I do that or not / precision on small patch will it be impacted?

        for idx in range(0, points_vox.shape[0], 2):
            p1, p2 = points_vox[idx], points_vox[idx+1] 
            if np.all(p1 >= 3) and np.all(p2 >= 3) and np.all(p1 < dim[0]- 3) and np.all(p2 < dim[0] - 3): # 3 voxels --> 1.2mm
                kept_points = np.append(kept_points, points_vox[idx: idx+2], axis=0)

            # check if there is an aneurysm not entirely included inside the volume
            p1_is_inside = np.all(p1 >= 0) and np.all(p1 < dim[0]) # p1 is inside the patch volume ?
            p2_is_inside = np.all(p2 >= 0) and np.all(p2 < dim[0]) # p2 is inside the patch volume ?
            if ignore == False and (p1_is_inside is not p2_is_inside):
                ignore = True
        
    return {'volume':v , 'vox2met': v2m, 'aneurysms': kept_points, 'fully_covered': not ignore, "residual": loss}

def getPatch(vol,vox2met,center,size,dim,affine=None,disp=None):
    '''
    Extract a 3D patch from a volume (vol) with faces parallel to some metric (in mm, e.g. RAS) frame coordinate directions. 
    vox2met is the 4x4 affine transformation between voxel and metric coordinates (a.k.a. IJKtoRAS transform)
    center (3-vector) is the center (in mm) of the patch to extract
    size (scalar or 3-vector) is size in mm of the patch to extract
    dim (int of 3-vector of ints) is the voxel dimension of the patch
    Optional arguments:
    affine is a 4X4 transform matrix to apply in the metric coordinate frame
        as a affine perturbation
    disp is 3-tuple of (x,y,z) displacements defining a spline transform
        the corners of the disp elements match those of the patch (e.g. 
        disp[:][0,0,0] provides the displacement to apply to patch[0,0,0]
        and disp[:][dd0-1,dd1-1,dd2-1] gives the displacement to apply to
        patch[dp0-1,dp1-1,dp2-1], assuming that dd is the dimension of the disp cube
        and dp is the dimension of the patch cube). A third order spline interpolation
        if performed on the displacements
    Returns the patch values, as well as the associated vox2met transform, without taking into account the possible
        affine transformation and distortion
    '''
    if np.isscalar(dim):
        dim = to_array(dim, length=3)
    if np.isscalar(size):
        size = to_array(size, length=3)

    center=np.asarray(center)
    dim=np.asarray(dim)
    size=np.asarray(size)

    #compute new vox2met transform associated with patch
    res = size/dim
    trans_patch = np.eye(4)
    trans_patch[:3,:3] = np.diag(res) # the patch axes are parallel to the metric space axes
    trans_patch[:3,3] = center - size/2 +res/2
    
    # RAS coordinates of voxels in the patch
    #   voxel coordinates (wrt patch), in homogeneous space
    p=np.vstack((np.mgrid[0:dim[0],0:dim[1],0:dim[2]].reshape((3,-1)),np.ones(np.prod(dim))))
    #   express the points in metric coordinates
    p=trans_patch@p
    # apply inverse transform
    p[:3]=invTransPoint(p[:3], center, size, affine, disp)
    
    # go back to voxel coordinates in the original volume
    D=np.linalg.inv(vox2met)
    c=(D@p)[:3]

    # note: could avoid computations by avoiding multiple transforms
    return sndi.map_coordinates(vol, c, order=1, prefilter=False).reshape(dim), trans_patch