import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import (Conv2D, AveragePooling2D)

from tensorflow.keras.initializers import GlorotUniform


# Defining network Below:
class DiscriminatorBlock(Layer):
    def __init__(self, img_dim, out_channels, kernel_size=1):
        super(DiscriminatorBlock, self).__init__()

        self.conv_0 = Conv2D(out_channels, kernel_size,
                             padding='same', activation='LeakyRelu')
        self.conv_1 = Conv2D(out_channels, kernel_size,
                             padding='same', activation='LeakyRelu')
        self.downsample = AveragePooling2D()

    def call(self, x, training=False):
        # Call layers of network on input x
        # Use the training variable to handle adding layers such as Dropout
        # and Batch Norm only during training
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.downsample(x)
        return x
