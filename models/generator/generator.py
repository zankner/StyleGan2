import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import (Dense, Flatten, Conv2D, Conv2DTranspose,
                                     BatchNormalization, concatenate,
                                     MaxPool2D)
from tensorflow.keras import Model
from models.generator import SynthesisBlock


#Defining network Below:
class Generator(Model):
    def __init__(self, z_dim):
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

        synthesis_network = []
        for block_dim in range(9):
            synthesis_network.append(
                SynthesisBlock(2**(block_dim), self.channels[i][0],
                               self.channels[i][1]))

    def call(self, x, training=False):
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

        gen_img = tf.random.normal(x.shape[0], 4, 4, 512)
        for i, synth_block in enumerate(self.synthesis_network):
            noise = tf.random.normal(x.shape[0], 2**i, 2**i,
                                     self.channels[i][1])
            gen_img = synth_block(gen_img, w, noise, i == 0)

        return gen_img
