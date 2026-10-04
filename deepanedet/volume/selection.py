import numpy as np
import sklearn.neighbors as skn
import scipy.ndimage as sndi


def extractPointsThres(vol, vox2met, thresLow, thresHigh):
    """
    return points in vol whose value is between thresLow (strictly)
    and thresHigh (loosely)
    return points coordinates (p) and values at these points (v)
    p is returned as a Nx3 array (or D is len(vol.shape) == D)
    and v is returned as 1xN-array
    """
    idx = np.nonzero(np.logical_and(vol > thresLow, vol <= thresHigh))
    v = vol[idx]
    p = vox2met[:3, :] @ np.vstack((idx, np.ones(len(v)))) + (0.5 * vox2met[:3, :3] @ np.ones(3)).reshape((-1, 1))

    return p.T, v


def selectPoints(vol, vox2met, thresLow, r, thresHigh=None, fPoints=None, nbPoints=None, extractType="Vessels"):
    """
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
    """
    if thresHigh is None:
        thresHigh = np.max(vol.ravel())
    p, v = extractPointsThres(vol, vox2met, thresLow, thresHigh)

    if (nbPoints is None) or (nbPoints > len(v)):
        nbPoints = len(v)

    if extractType == "Vessels":
        order = np.argsort(v)[::-1]  # pick points with decreasing voxel values
    else:
        order = np.arange(len(v))
        np.random.shuffle(order)  # pick points randomly
    removed = np.zeros(len(v), dtype=np.bool)
    tree = skn.KDTree(p)

    if fPoints is not None and len(fPoints) > 0:
        if len(fPoints.shape) == 1:
            fPoints = fPoints[np.newaxis, :]
        for i in tree.query_radius(fPoints, r):
            removed[i] = True

    ret = np.empty((0, 3))

    n = 0
    for idx in order:
        if not removed[idx]:
            q = p[idx, :].copy()
            i = tree.query_radius(q[np.newaxis, :], r)[0]
            removed[i] = True
            ret = np.vstack((ret, q[np.newaxis, :]))
            n = n + 1
            if n == nbPoints:
                return ret
    return ret


def getBall(r):
    """
    return a structure element shaped as a ball of radius r.
    can be used with skimage.morphology operators
    """
    x, y, z = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
    return ((x * x + y * y + z * z) <= r * r).astype(np.uint8)


def fillBetweenEdges(edges):
    """
    For each line along the first axis, fills every voxel between its first and last edge voxels (inclusive).
    edges: boolean volume; returns a uint8 mask of the same shape.
    """
    has_edge = edges.any(axis=0)
    first = np.argmax(edges, axis=0)
    last = edges.shape[0] - 1 - np.argmax(edges[::-1], axis=0)
    x = np.arange(edges.shape[0]).reshape((-1,) + (1,) * (edges.ndim - 1))
    return ((x >= first) & (x <= last) & has_edge).astype(np.uint8)


def removeSkullMask(vol, percent=80):
    """
    Skull Stripping operation: the volume between the outermost strong edges of each line (gradient
    magnitude above its percent-th percentile), eroded to remove the skull.
    """
    edges = sndi.gaussian_gradient_magnitude(vol, sigma=3)
    mask = fillBetweenEdges(edges >= np.percentile(edges, percent))

    # erode this mask to remove the skull
    selem = getBall(2)
    for _ in range(15):
        mask = sndi.binary_erosion(mask, structure=selem, border_value=True).astype(np.uint8)
    # return this mask
    return mask
