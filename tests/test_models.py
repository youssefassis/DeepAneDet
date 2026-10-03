import pytest

from deepanedet.models.building_blocks import ExtResNetBlock


def test_resnet_block_without_activation_raises():
    with pytest.raises(ValueError, match="no activation"):
        ExtResNetBlock(1, 2, order="cb")
