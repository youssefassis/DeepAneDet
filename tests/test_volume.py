import numpy as np

import Volume.Selection as vs


def test_selected_points_avoid_every_forbidden_point():
    vol = np.ones((40, 1, 1))  # a line of 40 bright voxels, 1mm apart
    forbidden = np.array([[0.5, 0.5, 0.5], [30.5, 0.5, 0.5]])

    points = vs.selectPoints(vol, np.eye(4), thresLow=0.5, r=3, fPoints=forbidden)

    distances = np.linalg.norm(points[:, np.newaxis] - forbidden[np.newaxis], axis=-1)
    assert len(points) > 0 and distances.min() > 3


def test_points_in_radius_of_several_queries():
    points = np.array([[0.0, 0, 0], [10, 0, 0], [20, 0, 0]])

    assert vs.pointsInRadius(np.array([[0.0, 0, 0], [20, 0, 0]]), points, 1).tolist() == [0, 2]
