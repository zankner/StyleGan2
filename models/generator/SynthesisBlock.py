import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import (Dense, Flatten, Conv2D, Conv2DTranspose,
                                     BatchNormalization, UpSampling2D,
                                     MaxPool2D, Layer)

from tensorflow.keras.initializers import GlorotUniform


# Defining network Below:
class SynthesisBlock(Layer):
    def __init__(self, channels, img_dim, kernel_size=3):
        super(SynthesisBlock, self).__init__()

        self.channels = channels
        self.img_dim = img_dim

        # Define layers of the network:
        self.upsample_0 = UpSampling2D()
        self.upsample_1 = UpSampling2D()

        self.y_0 = Dense(channels)
        self.y_1 = Dense(channels)
        self.y_2 = Dense(channels)
        self.y_3 = Dense(channels)

        self.xavier = GlorotUniform()

        conv_shape = (3, 3, channels, channels)
        self.conv_0 = tf.Variable(initializer=self.xavier(shape=conv_shape))
        self.conv_1 = tf.Variable(initializer=self.xavier(shape=conv_shape))
        self.conv_2 = tf.Variable(initializer=self.xavier(shape=conv_shape))
        self.conv_3 = tf.Variable(initializer=self.xavier(shape=conv_shape))

    def build(self, input_shape):
        noise_scale_shape = (input_shape[0], self.img_dim, self.img_dim, 1)
        self.noise_scale_0 = self.add_weight(shape=noise_scale_shape,
                                             initializer='glorot_uniform')
        self.noise_scale_1 = self.add_weight(shape=noise_scale_shape,
                                             initializer='glorot_uniform')
        self.noise_scale_2 = self.add_weight(shape=noise_scale_shape,
                                             initializer='glorot_uniform')
        self.noise_scale_3 = self.add_weight(shape=noise_scale_shape,
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
