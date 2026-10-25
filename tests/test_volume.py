import numpy as np

from deepanedet.volume import selection as vs


def test_selected_points_avoid_every_forbidden_point():
    vol = np.ones((40, 1, 1))  # a line of 40 bright voxels, 1mm apart
    forbidden = np.array([[0.5, 0.5, 0.5], [30.5, 0.5, 0.5]])

    points = vs.select_points(vol, np.eye(4), thres_low=0.5, r=3, forbidden_points=forbidden)

    distances = np.linalg.norm(points[:, np.newaxis] - forbidden[np.newaxis], axis=-1)
    assert len(points) > 0 and distances.min() > 3


def test_fill_between_edges_fills_each_line_inclusively():
    edges = np.zeros((10, 3, 2), dtype=bool)
    edges[[3, 7], 0, 0] = True  # line with two edges
    edges[5, 1, 0] = True  # line with a single edge voxel
    edges[[0, 9], 2, 1] = True  # last line, spanning the whole axis

    mask = vs.fill_between_edges(edges)

    assert np.flatnonzero(mask[:, 0, 0]).tolist() == [3, 4, 5, 6, 7]
    assert np.flatnonzero(mask[:, 1, 0]).tolist() == [5]
    assert mask[:, 2, 1].all()
    assert mask.sum() == 5 + 1 + 10  # untouched lines stay empty


def test_skull_mask_of_a_symmetric_head_is_symmetric():
    x, y, z = np.ogrid[-40:41, -40:41, -40:41]
    radius = np.sqrt(x * x + y * y + z * z)
    head = np.where(radius < 30, 1.0, 0.0) + np.where((radius >= 30) & (radius < 34), 3.0, 0.0)  # brain + skull

    mask = vs.remove_skull_mask(head)

    assert mask[40, 40, 40] == 1 and mask[40, 40, 72] == 0  # center kept, skull removed
    np.testing.assert_array_equal(mask, mask[::-1])


def test_selected_points_are_voxel_centers_consistent_with_patch_sampling():
    import pytest

    from deepanedet.volume.patch import get_patch

    vox2met = np.array([[0.4, 0, 0, -10.0], [0, 0.4, 0, 5.0], [0, 0, 0.6, 2.0], [0, 0, 0, 1]])
    vol = np.zeros((30, 30, 30))
    vol[12, 7, 20] = 1.0

    (point,) = vs.select_points(vol, vox2met, thres_low=0.5, r=1)

    np.testing.assert_allclose(point, (vox2met @ [12, 7, 20, 1])[:3])
    patch, _ = get_patch(vol, vox2met, point, size=[1.2, 1.2, 1.8], dim=[3, 3, 3])  # 3 voxels at the volume resolution
    assert patch[1, 1, 1] == pytest.approx(1.0)
