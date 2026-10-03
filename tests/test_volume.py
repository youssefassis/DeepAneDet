import numpy as np

from deepanedet.volume import selection as vs


def test_selected_points_avoid_every_forbidden_point():
    vol = np.ones((40, 1, 1))  # a line of 40 bright voxels, 1mm apart
    forbidden = np.array([[0.5, 0.5, 0.5], [30.5, 0.5, 0.5]])

    points = vs.selectPoints(vol, np.eye(4), thresLow=0.5, r=3, fPoints=forbidden)

    distances = np.linalg.norm(points[:, np.newaxis] - forbidden[np.newaxis], axis=-1)
    assert len(points) > 0 and distances.min() > 3


def test_fill_between_edges_fills_each_line_inclusively():
    edges = np.zeros((10, 3, 2), dtype=bool)
    edges[[3, 7], 0, 0] = True  # line with two edges
    edges[5, 1, 0] = True  # line with a single edge voxel
    edges[[0, 9], 2, 1] = True  # last line, spanning the whole axis

    mask = vs.fillBetweenEdges(edges)

    assert np.flatnonzero(mask[:, 0, 0]).tolist() == [3, 4, 5, 6, 7]
    assert np.flatnonzero(mask[:, 1, 0]).tolist() == [5]
    assert mask[:, 2, 1].all()
    assert mask.sum() == 5 + 1 + 10  # untouched lines stay empty


def test_skull_mask_of_a_symmetric_head_is_symmetric():
    x, y, z = np.ogrid[-40:41, -40:41, -40:41]
    radius = np.sqrt(x * x + y * y + z * z)
    head = np.where(radius < 30, 1.0, 0.0) + np.where((radius >= 30) & (radius < 34), 3.0, 0.0)  # brain + skull

    mask = vs.removeSkullMask(head)

    assert mask[40, 40, 40] == 1 and mask[40, 40, 72] == 0  # center kept, skull removed
    np.testing.assert_array_equal(mask, mask[::-1])
