import numpy as np
import pytest
import torch

import prediction


class OneDetectionPerPatch(torch.nn.Module):
    """Detects a sphere in the first cell of a 2x2x2 grid of every patch."""

    def forward(self, x):
        confidence = torch.full((x.shape[0], 2, 2, 2, 1), -10.0)
        confidence[:, 0, 0, 0, 0] = 10.0
        return [[confidence, torch.zeros((x.shape[0], 2, 2, 2, 4))]]


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_patch_wise_prediction_does_not_depend_on_batch_size(batch_size):
    volume = np.zeros((16, 8, 8), dtype=np.float32)  # two 8-voxel patches along x

    _, spheres = prediction.ndl_patch_wise_prediction(
        "cpu", OneDetectionPerPatch(), volume, [8, 8, 8], iou_threshold=0.01, detections_per_patch=None, batch_size=batch_size
    )

    # cell centers (offset sigmoid(0) = 0.5 of a 4-voxel cell) in each patch
    assert sorted(s[:3] for s in spheres) == [[2.0, 2.0, 2.0], [10.0, 2.0, 2.0]]
