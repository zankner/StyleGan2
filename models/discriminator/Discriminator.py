import numpy as np
import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import (Dense, Flatten, Conv2D, AveragePooling2D)
from tensorflow.keras import Model
from models.generator import SynthesisBlock


# Defining network Below:
class Discriminator(Model):
    def __init__(self, fmap_base=16 << 10, fmap_decay=1.0,
                 fmap_min=1, fmap_max=512, kernel_size=3):
        super(Discriminator, self).__init__()
        # Define layers of the network:
        self.dense_0 = Dense(2, activation='softmax')

        feat_maps = []
        for stage in range(9):
            feat_maps.append(
                np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))),
                        fmap_min, fmap_max))
        feat_maps.reverse()

        self.discriminator_network = []
        for i in range(9):
            discriminator_block = []
            discriminator_block.append(
                Conv2D(feat_maps[i], kernel_size, padding='same'))
            discriminator_block.append(AveragePooling2D())
            self.discriminator_network.append(discriminator_block)

        self.from_rgb = []
        for i in range(9):
            self.from_rgb.append([
                Conv2D(feat_maps[i], kernel_size, padding='same'), AveragePooling2D()])

    def call(self, x, progressive_depth, alpha, training=False):
        # Call layers of network on input x
        # Use the training variable to handle adding layers such as Dropout
        # and Batch Norm only during training
        res_block = None

        for i in range(progressive_depth):
            if i == 0 and progressive_depth > 1:
                res_scaled = self.from_rgb[progressive_depth - 1][1](x)
                res_block = self.from_rgb[progressive_depth -
                                          1][0](res_scaled)
            x = self.discriminator_network[progressive_depth - (i + 1)][0](x)
            if i < progressive_depth - 1:
                x = self.discriminator_network[progressive_depth -
                                               (i + 1)][1](x)
            if i == 0 and progressive_depth > 1:
                x = ((1 - alpha) * x) + ((1 - alpha) * res_block)
        x = self.dense_0(x)

        return x
