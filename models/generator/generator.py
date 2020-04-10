import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import (Dense, Conv2D, UpSampling2D)
from tensorflow.keras import Model
from models.generator import SynthesisBlock


# Defining network Below:
class Generator(Model):
    def __init__(self, z_dim, progressive_depth):
        super(Generator, self).__init__()
        # Define layers of the network:
        self.map_net_1 = Dense(z_dim)
        self.map_net_2 = Dense(z_dim)
        self.map_net_3 = Dense(z_dim)
        self.map_net_4 = Dense(z_dim)
        self.map_net_5 = Dense(z_dim)
        self.map_net_6 = Dense(z_dim)
        self.map_net_7 = Dense(z_dim)
        self.map_net_8 = Dense(z_dim)

        self.channels = [[512, 512], [512, 512], [512, 512], [512, 512],
                         [512, 256], [256, 128], [128, 64], [64, 32], [32, 16]]

        self.synthesis_network = []
        for block_dim in range(9):
            self.synthesis_network.append(
                SynthesisBlock(2**(block_dim), self.channels[block_dim][0],
                               self.channels[block_dim][1]))

        self.to_rbg = []
        for rbg_dim in range(9):
            to_rgb_block = []
            to_rgb_block.append(UpSampling2D(interpolation='bilinear'))
            to_rgb_block.append(Conv2D(3, 1, padding='same'))
            self.to_rbg.append(to_rgb_block)

    def call(self, x, progressive_depth, alpha, training=False):
        # Call layers of network on input x
        # Use the training variable to handle adding layers such as Dropout
        # and Batch Norm only during training
        x = self.map_net_1(x)
        x = self.map_net_2(x)
        x = self.map_net_3(x)
        x = self.map_net_4(x)
        x = self.map_net_5(x)
        x = self.map_net_6(x)
        x = self.map_net_7(x)
        w = self.map_net_8(x)

        res_block = None
        gen_img = tf.random.normal(x.shape[0], 4, 4, 512)
        for i in range(progressive_depth):
            noise = tf.random.normal(x.shape[0], 2**i, 2**i,
                                     self.channels[i][1])
            gen_img = self.synthesis_network[i](gen_img, w, noise, i == 0)

            if progressive_depth > 1:
                if i == progressive_depth - 2:
                    gen_img = self.to_rbg[i - 1][0](gen_img)
                    res_block = self.to_rbg[i - 1][1](gen_img)

            if i == progressive_depth - 1:
                gen_img = self.to_rbg[i](
                    (gen_img * alpha) + (res_block * (1 - alpha)))

        return gen_img
