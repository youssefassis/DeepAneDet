import torch.nn as nn

def number_of_features_per_level(init_channel_number, num_levels):
    '''
    This is a simple function that calculates the number of features per level in a neural network.
    It takes two arguments:
        - init_channel_number: an integer that represents the number of channels in the first layer of the network.
        - num_levels: an integer that represents the total number of levels in the network.
    
    The function returns a list containing the number of features for each level in the network.
    The number of features at each level is calculated by multiplying the init_channel_number 
    with 2 raised to the power of the level number. For example, if init_channel_number is 16
    and num_levels is 4, the function will return the list [16, 32, 64, 128], indicating that
    the number of features will double at each level.
   '''
    return [init_channel_number * 2 ** k for k in range(num_levels)]

def createBN (in_channels, eps=1e-5, affine=True):
    '''
    This function creates a batch normalization layer using PyTorch nn.BatchNorm3d module.
    It takes in the number of input channels and two arguments:
        - eps (default=1e-5) is the value added to the denominator of the batch 
        normalization expression to improve numerical stability. 
        - affine (default=True) is a boolean value that determines whether to apply an affine 
        transformation after normalization. If True, the module will learn two
        parameters per input channel: a scaling parameter and a bias parameter.
    '''
    return nn.BatchNorm3d(in_channels, eps = eps, affine = affine)

def createLeakyReLU (negative_slope=0.01, inplace=True):
    '''
    This function creates a LeakyReLU activation function with the given negative slope and inplace parameters.
    '''
    return nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)


def createConv(inp_feat, out_feat, kernel=3, stride=1, padding=1, bias=True):
    '''
    This function creates a batch normalization layer using PyTorch nn.BatchNorm3d module.
    It takes in the number of input channels and two arguments:
        - eps (default=1e-5) is the value added to the denominator of the batch 
        normalization expression to improve numerical stability. 
        - affine (default=True) is a boolean value that determines whether to apply an affine 
        transformation after normalization. If True, the module will learn two
        parameters per input channel: a scaling parameter and a bias parameter.
    '''
    return nn.Conv3d(inp_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=bias)

def create_conv(in_channels, out_channels, kernel_size, order, padding, stride=1, dropout=False):
    '''
    This function creates a convolutional block with different options for the order of operations.
    It takes in the number of input and output channels, kernel size, order of operations, padding, and stride as arguments.
    The order of operations is specified using a string where each character corresponds to a specific operation:
        - 'c': convolutional layer
        - 'r': ReLU activation function
        - 'l': LeakyReLU activation function
        - 'm': Mish activation function
        - 'b': batch normalization layer
        - 'i': instance normalization layer
        - 'd': dropout layer
    The function checks that a convolutional layer is present in the order since this is required. It also checks 
    that a non-linearity is not the first operation in the layer.
    The function returns a list of nn.Module objects corresponding to the specified order of operations. 
        
    '''
    assert 'c' in order, "Conv layer MUST be present in each block"
    assert order[0] not in 'rlm', 'Non-linearity cannot be the first operation in the layer'
    modules = []
    for i, char in enumerate(order):
        # Activation functions
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', createLeakyReLU()))
        elif char == 'm':
            modules.append(('Mish', nn.Mish(inplace=True)))
            
        # Convolutions
        elif char == 'c':
            bias = not ('b' in order)
            modules.append(('conv', createConv(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)))
        
        elif char == 'd':
            modules.append(('dropout', nn.Dropout(p=0.1, inplace=True)))
            
        # Normalzation layers
        elif char == 'b':
            is_before_conv = (i < order.find('c'))
            if is_before_conv:
                modules.append(('batchnorm', createBN(in_channels)))
            else:
                modules.append(('batchnorm', createBN(out_channels)))
        elif char == 'i':
            is_before_conv = (i < order.find('c'))
            if is_before_conv:
                modules.append(('instancenorm', nn.InstanceNorm3d(in_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False)))
            else:
                modules.append(('instancenorm', nn.InstanceNorm3d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['r', 'd', 'm', 'l', 'c', 'b', 'i']")
    return modules

class SingleConv(nn.Sequential):
    '''
    The SingleConv class is a subclass of nn.Sequential and represents a single convolutional block in a neural network.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, order='cbr', padding=1, stride=1, dropout=False):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order,
                                        padding=padding, stride=stride, dropout=dropout):
            self.add_module(name, module)

class DoubleConv(nn.Sequential):
    '''
    The DoubleConv class is a subclass of nn.Sequential and contains two successive convolutional blocks.
    '''
    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='cbr', padding=1, stride=1, dropout=False):
        super(DoubleConv, self).__init__()
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels
        self.add_module('SingleConv1', SingleConv(conv1_in_channels, conv1_out_channels, 
                                                  kernel_size, order, stride=stride, padding=padding))
        self.add_module('SingleConv2', SingleConv(conv2_in_channels, conv2_out_channels, 
                                                  kernel_size, order, padding=padding, dropout=dropout))

        
class ExtResNetBlock(nn.Module):
    '''
    The ExtResNetBlock class represents a residual block containing three successive convolutional blocks.
    It applies the first convolution and saves its output as a residual. The residual block is then 
    applied to the output of the first convolution. The output of the residual block is then added to the
    residual, and the non-linearity is applied to the result. Finally, the output is returned.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, order='cbr', padding=1, stride=1,  **kwargs):
        super(ExtResNetBlock, self).__init__()

        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, stride=stride, padding=padding)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, padding=padding)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order, padding=padding)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'm' in order:
            self.non_linearity = nn.Mish( inplace=True)
        elif 'r' in order:
            self.non_linearity = nn.ReLU( inplace=True)
        else:
            raise ValueError(f"Layer order '{order}' has no activation function ('l', 'm' or 'r')")

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out
        # residual block
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.non_linearity(out)
        return out        
        
class Encoder(nn.Module):
    '''
    The Encoder class represents an encoder block in a 3D convolutional neural network. It applies 
    a sequence of convolutional blocks followed by a down sampling operation to reduce the spatial
    dimensions of the input tensor.
    Parameters:
        - in_channels: the number of input channels to the block.
        - out_channels: the number of output channels from the block.
        - conv_kernel_size: the size of the convolution kernel.
        - apply_pooling: a boolean indicating whether to apply pooling after the convolutional layers (default False).
        - pool_kernel_size: the size of the pooling kernel (default 2).
        - pool_type: the type of pooling to apply, either 'max', 'avg', or 'conv' (default 'max').
        - basic_module: the type of convolutional block to use, e.g. DoubleConv or ExtResNetBlock (default DoubleConv).
        - conv_layer_order: the order of convolutional layers to use (default 'gcr').
        - padding: the amount of zero-padding to apply (default 1).
    '''
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=False,
                 pool_kernel_size=2, pool_type='max', basic_module=DoubleConv,
                 conv_layer_order='gcr', padding=1, dropout=False):
        super(Encoder, self).__init__()

        assert pool_type in ['max', 'avg', 'conv']
        
        self.pooling = None
        stride = 1
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            elif pool_type == 'avg':
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
            else:
                stride = pool_kernel_size # = 2
        
        self.basic_module = basic_module(in_channels, 
                                         out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         padding=padding,
                                         stride=stride,
                                         dropout=dropout)
    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x

def create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, pool_kernel_size, pool_type, apply_pooling=False):
    '''
    This function creates a list of encoder modules. 
    It takes the following arguments:
        - in_channels: the number of input channels
        - f_maps: a list of integers, where each integer specifies the number of output feature maps for an encoder module
        - basic_module: a PyTorch module that is used as the basic building block for the encoder
        - conv_kernel_size: the size of the convolutional kernel
        - conv_padding: the amount of padding to apply to the convolutional layer
        - layer_order: a string that specifies the order of the convolutional and pooling layers in the encoder module
        - pool_kernel_size: the size of the pooling kernel
        - pool_type: the type of pooling to use (e.g. max pooling or average pooling)
        - apply_pooling: a boolean that specifies whether to apply pooling after the first encoder module
    '''
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i != 0:
            in_channels = f_maps[i-1]
            apply_pooling = True
        encoder = Encoder(in_channels,
                          out_feature_num,
                          apply_pooling=apply_pooling,
                          basic_module=basic_module,
                          conv_layer_order=layer_order,
                          pool_kernel_size=pool_kernel_size,
                          conv_kernel_size=conv_kernel_size,
                          padding=conv_padding,
                          pool_type = pool_type)

        encoders.append(encoder)
    return nn.ModuleList(encoders)
