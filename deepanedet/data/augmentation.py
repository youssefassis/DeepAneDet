import numpy as np


def randomFlip(data, patch_shape, axes=[0], flip_probability=0):
    """This function flips image or image coordinates along the specified axis/axes with a given probability.
    data: expected to be list of 3D images or image coordinates.
    patch_shape: a list that specifies the shape of the data to flip.
    axes: a list of integers that specifies the axis/axes along which to flip the data.
    flip_probability: a float between 0 and 1 that specifies the probability of flipping the data.

    The function first performs some input validation to ensure that the axes parameter is not empty and that
    the flip_probability parameter is within the valid range of [0,1]. It then loops through each axis specified
    in the axes parameter and generates a random number between 0 and 1 to determine whether to flip the data
    along that axis. If the random number is less than or equal to the flip_probability, the data is flipped
    along that axis.
    """
    assert len(axes) > 0, "Flip around at least one axis"
    assert flip_probability >= 0 and flip_probability <= 1, "Flip probability should be positive"

    for axis in axes:
        prob = np.random.uniform()
        if prob <= flip_probability:
            for idx in range(len(data)):
                assert data[idx].ndim in [2, 3], "Supports only 3D (DxHxW) images"
                if data[idx].ndim == 3:  # volume
                    data[idx] = np.flip(data[idx], axis).copy()
                elif data[idx].ndim == 2:  # Coordinates
                    data[idx][:, axis] = np.abs(patch_shape[axis] - data[idx][:, axis] - 1)
    return data
