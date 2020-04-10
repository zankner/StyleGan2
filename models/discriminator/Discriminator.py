import numpy as np
import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import (Dense, Flatten, Conv2D, AveragePooling2D)
from tensorflow.keras import Model
from models.generator import SynthesisBlock


# Defining network Below:
class Discriminator(Model):
    def __init__(self, fmap_base, fmap_decay, fmap_min, fmap_max, kernel_size):
        super(Discriminator, self).__init__()
        # Define layers of the network:
        feat_maps = [np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))),
                             fmap_min, fmap_max) for stage in range(9)].reverse()

        self.discriminator_network = []
        for i in range(9):
            self.discriminator_network.append([
                Conv2D(feat_maps[i], kernel_size, padding='same'), AveragePooling2D()])

        self.from_rgb = []
        for rbg_dim in range(9):
            self.from_rgb.append([
                Conv2D(feat_maps[i], kernel_size, padding='same'), AveragePooling2D()])

    def call(self, x, progressive_depth, alpha, training=False):
        # Call layers of network on input x
        # Use the training variable to handle adding layers such as Dropout
        # and Batch Norm only during training
        res_block = None

        for i in range(progressive_depth):
            if i == 0:
                res_scaled = self.from_rgb[progressive_depth - 1][1](x)
                res_block = self.from_rgb[progressive_depth -
                                          1][0](res_scaled)
                x = self.from_rgb[progressive_depth][1](x)
            x = self.discriminator_network[progressive_depth - i][0](x)
            x = self.discriminator_network[progressive_depth - i][1](x)
            x = ((1 - alpha) * x) + ((1 - alpha) * res_block)
