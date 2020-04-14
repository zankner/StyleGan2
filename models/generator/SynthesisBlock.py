import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import (Dense, Flatten, Conv2D, Conv2DTranspose,
                                     BatchNormalization, UpSampling2D,
                                     MaxPool2D, Layer)

from tensorflow.keras.initializers import GlorotUniform


# Defining network Below:
class SynthesisBlock(Layer):
    def __init__(self, img_dim, in_channels, out_channels, initial=False, 
            kernel_size=3):
        super(SynthesisBlock, self).__init__()

        self.img_dim = img_dim if initial else img_dim * 2

        self.out_channels = out_channels

        # Define layers of the network:
        self.upsample_0 = UpSampling2D(interpolation='bilinear')

        self.y_0 = Dense(out_channels)
        self.y_1 = Dense(out_channels)

        conv_shape = (kernel_size, kernel_size, in_channels, out_channels)
        self.conv_0 = self.add_weight(
            shape=conv_shape, initializer='glorot_uniform')
        self.conv_1 = self.add_weight(
            shape=conv_shape, initializer='glorot_uniform')

    def build(self, input_shape):
        noise_scale_shape = (input_shape[0], self.img_dim, self.img_dim,
                             self.out_channels)
        self.noise_scale_0 = self.add_weight(shape=noise_scale_shape,
                                             initializer='glorot_uniform')
        self.noise_scale_1 = self.add_weight(shape=noise_scale_shape,
                                             initializer='glorot_uniform')
        
        bias_shape = (input_shape[0], self.img_dim, self.img_dim, 
                self.out_channels)
        self.bias_0 = self.add_weight(shape=bias_shape,
                initializer='glorot_uniform')
        self.bias_1 = self.add_weight(shape=bias_shape,
                initializer='glorot_uniform')

    def call(self, x, w, noise, initial=False, training=False):
        # Call layers of network on input x
        # Use the training variable to handle adding layers such as Dropout
        # and Batch Norm only during training
        if not initial:
            x = self.upsample_0(x)

        s = self.y_0(w)
        s = tf.reshape(s, [s.shape[0], 1, 1, 1, s.shape[1]])
        w_prime = self.conv_0 * s
        std = tf.math.rsqrt(
            tf.math.reduce_sum(tf.square(w_prime), axis=[1, 2, 3]) + 1e-8)
        std = tf.reshape(std, [std.shape[0], 1, 1, 1, std.shape[1]])
        w_prime = w_prime * std

        splits = [1 for i in range(x.shape[0])]
        split_x = tf.split(x, splits)
        split_weights = tf.split(w_prime, splits)
        
        mod_conv_x = []
        for input_, weight in zip(split_x, split_weights):
            weight = tf.reshape(weight, [weight.shape[1], weight.shape[2],
                weight.shape[3], weight.shape[4]])
            mod_conv_x.append(tf.nn.conv2d(input_, weight, 1, 'SAME'))
            
        x = tf.concat(mod_conv_x, 0)

        x = x + (self.noise_scale_0 * noise)

        x += self.bias_0

        x = tf.nn.leaky_relu(x)

        if not initial:
            s = self.y_1(w)
            s = tf.reshape(s, [s.shape[0], 1, 1, 1, s.shape[1]])
            w_prime = self.conv_1 * s
            std = tf.math.rsqrt(
                tf.math.reduce_sum(tf.square(w_prime), axis=[1, 2, 3]) + 1e-8)
            std = tf.reshape(std, [std.shape[0], 1, 1, 1, std.shape[1]])
            w_prime = w_prime * std

            splits = [1 for i in range(x.shape[0])]
            split_x = tf.split(x, splits)
            split_weights = tf.split(w_prime, splits)
            
            mod_conv_x = []
            for input_, weight in zip(split_x, split_weights):
                weight = tf.reshape(weight, [weight.shape[1], weight.shape[2],
                    weight.shape[3], weight.shape[4]])
                mod_conv_x.append(tf.nn.conv2d(input_, weight, 1, 'SAME'))
                

            x = x + (self.noise_scale_1 * noise)

            x += self.bias_1

            x = tf.nn.leaky_relu(x)

        return x


s = SynthesisBlock(4, 512, 512)
w = tf.random.uniform([3, 512])
x = tf.random.uniform([3, 4, 4, 512])
noise = tf.random.uniform([1, 1, 1, 512])
g = s(x, w, noise)
print(g.shape)
