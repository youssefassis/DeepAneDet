import torch.nn as nn
from deepanedet.models.building_blocks import number_of_features_per_level, create_encoders, DoubleConv


def createBN(in_channels, eps=1e-5, affine=True):
    """
    This function creates a batch normalization layer using PyTorch nn.BatchNorm3d module.
    It takes in the number of input channels and two arguments:
        - eps (default=1e-5) is the value added to the denominator of the batch
        normalization expression to improve numerical stability.
        - affine (default=True) is a boolean value that determines whether to apply an affine
        transformation after normalization. If True, the module will learn two
        parameters per input channel: a scaling parameter and a bias parameter.
    """
    return nn.BatchNorm3d(in_channels, eps=eps, affine=affine)


class CNNBlock(nn.Module):
    """
    This is a Module for a CNN block with a 3D convolutional layer, batch normalization, and
    LeakyReLU activation. Used as a building block for the Detection Head.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))


class DetectionHead(nn.Module):
    """
    This is a module that defines a detection block used in object detection networks.
    The detection block takes an input tensor x of shape (batch_size, in_channels, height, width, depth),
    where in_channels is the number of input channels, and height, width, and depth are the spatial dimensions
    of the input tensor.
    The module has the following attributes:
        - params_per_element: an integer that specifies the number of parameters for each object detection element
        (e.g. the x-coordinate, y-coordinate, z-coordinate, width, height, depth, and confidence score)
        - block: a DoubleConv module that applies two convolutional layers to the input tensor and downsamples the spatial dimensions by a factor of 2
        - conv_confidence: a convolutional layer that outputs the confidence scores
        - conv_regression1: a convolutional layer that outputs the regression values for the center coordinates
        - conv_regression2: a convolutional layer that outputs the regression values for the width, height and depth
    """

    def __init__(self, in_channels, params_per_box, anchors_per_scale):
        super(DetectionHead, self).__init__()
        self.params_per_box = params_per_box
        self.pred = nn.Sequential(
            CNNBlock(in_channels, in_channels // 2, kernel_size=3, padding=1),
            CNNBlock(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
        )
        self.anchors_per_scale = anchors_per_scale
        self.params_per_box = params_per_box

        self.conv_confidence = nn.Conv3d(in_channels // 2, 1 * self.anchors_per_scale, kernel_size=1, bias=True)
        self.conv_regression = nn.Conv3d(
            in_channels // 2, params_per_box * self.anchors_per_scale, kernel_size=1, bias=True
        )

    def forward(self, x):
        x = self.pred(x)
        if self.anchors_per_scale == 1:
            x1 = (
                self.conv_confidence(x)
                .reshape(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4])
                .permute(0, 2, 3, 4, 1)
                .contiguous()
            )  # confidence
            x2 = (
                self.conv_regression(x)
                .reshape(x.shape[0], self.params_per_box, x.shape[2], x.shape[3], x.shape[4])
                .permute(0, 2, 3, 4, 1)
                .contiguous()
            )  # regression parameters
        else:
            x1 = (
                self.conv_confidence(x)
                .reshape(x.shape[0], self.anchors_per_scale, 1, x.shape[2], x.shape[3], x.shape[4])
                .permute(0, 1, 3, 4, 5, 2)
                .contiguous()
            )  # confidence
            x2 = (
                self.conv_regression(x)
                .reshape(x.shape[0], self.anchors_per_scale, self.params_per_box, x.shape[2], x.shape[3], x.shape[4])
                .permute(0, 1, 3, 4, 5, 2)
                .contiguous()
            )  # regression parameters
        return [x1, x2]


class AssisModel(nn.Module):
    """
    This is a YOLO-based neural network for 3D object detection using spherical representation.
    The network takes a 3D input image with a 'single' channel and feeds it through a series of
    convolutional layers with pooling to extract hierarchical features.
    The create_encoders function generates the encoder layers. The DetectionHead class defines a
    detection block with convolutional layers to predict (i.e. detect aneurysms) the spheres.
    """

    def __init__(
        self,
        in_channels=1,
        f_maps=64,
        layer_order="cbl",
        pool_type="conv",
        num_levels=4,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
        basic_module=DoubleConv,
        params_per_element=4,
        anchors_per_scale=1,
        nb_scales=1,
    ):
        super(AssisModel, self).__init__()
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(init_channel_number=f_maps, num_levels=num_levels)
        assert isinstance(f_maps, list)
        assert len(f_maps) > 1, "Required at least 2 levels"

        self.encoders = create_encoders(
            in_channels=in_channels,
            f_maps=f_maps,
            basic_module=basic_module,
            conv_kernel_size=conv_kernel_size,
            conv_padding=conv_padding,
            layer_order=layer_order,
            pool_kernel_size=pool_kernel_size,
            pool_type=pool_type,
        )
        self.ScaleDetections = nn.ModuleList(
            [
                DetectionHead(in_channels=i, params_per_box=params_per_element, anchors_per_scale=anchors_per_scale)
                for i in list(reversed(f_maps[-nb_scales:]))
            ]
        )

    def forward(self, x):
        """
        This forward method of the AssisModel class takes an input tensor x and passes it through
        the encoder layers to extract features. The features are then passed through the detection block
        to predict the object location and orientation, and the output is returned as a list of predictions.
        """
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
        detections = self.ScaleDetections[0](x)
        return [detections]
