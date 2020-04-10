import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import (Dense, Flatten, Conv2D, Conv2DTranspose,
                                     BatchNormalization, UpSampling2D,
                                     MaxPool2D, Layer)

from tensorflow.keras.initializers import GlorotUniform


# Defining network Below:
class SynthesisBlock(Layer):
    def __init__(self, img_dim, in_channels, out_channels, kernel_size=3):
        super(SynthesisBlock, self).__init__()

        self.img_dim = img_dim
        self.out_channels = out_channels

        # Define layers of the network:
        self.upsample_0 = UpSampling2D(interpolation='bilinear')

        self.y_0 = Dense(out_channels)
        self.y_1 = Dense(out_channels)

        self.xavier = GlorotUniform()

        conv_shape = (kernel_size, kernel_size, in_channels, out_channels)
        self.conv_0 = tf.Variable(initializer=self.xavier(shape=conv_shape))
        self.conv_1 = tf.Variable(initializer=self.xavier(shape=conv_shape))

    def build(self, input_shape):
        noise_scale_shape = (input_shape[0], self.img_dim, self.img_dim,
                             self.out_channels)
        self.noise_scale_0 = self.add_weight(shape=noise_scale_shape,
                                             initializer='glorot_uniform')
        self.noise_scale_1 = self.add_weight(shape=noise_scale_shape,
                                             initializer='glorot_uniform')

    def call(self, x, w, noise, initial=False, training=False):
        # Call layers of network on input x
        # Use the training variable to handle adding layers such as Dropout
        # and Batch Norm only during training
        if not initial:
            x = self.upsample_0(x)

        s = self.y_0(w)
        w_prime = self.conv_0 * s
        std = tf.math.rsqrt(
            tf.math.reduce_sum(tf.square(w_prime), axis=[1, 2, 3]) + 1e-8)
        w_prime = tf.transpose(tf.transpose(w_prime) * std)

        x = tf.nn.conv2d(x, w_prime)

        x = x + (self.noise_scale_0 * noise)

        x += self.bias_0

        x = tf.nn.leaky_relu(x)

        if not initial:
            s = self.y_1(w)
            w_prime = self.conv_1 * s
            std = tf.math.rsqrt(
                tf.math.reduce_sum(tf.square(w_prime), axis=[1, 2, 3]) + 1e-8)
            w_prime = tf.transpose(tf.transpose(w_prime) * std)

            x = tf.nn.conv2d(x, w_prime)

            x = x + (self.noise_scale_1 * noise)

            x += self.bias_0

            x = tf.nn.leaky_relu(x)

        return x
