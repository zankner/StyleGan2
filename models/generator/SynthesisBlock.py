import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, Conv2DTranspose, 
    BatchNormalization, concatenate, MaxPool2D, Layer
)

#Defining network Below:
class SynthesisBlock(Layer):
  def __init__(self, channels, img_dim, kernel_size = 3):
    super(Synthesis_block, self).__init__()
    
    self.channels = channels
    self.img_dim = img_dim

    # Define layers of the network:
    self.upsample = UpSampling2D()

    self.y_0 = Dense(channels * 2)
    self.y_1 = Dense(channels * 2)
    self.y_2 = Dense(channels * 2)
    self.y_3 = Dense(channels * 2)
    
    self.conv_0 = Conv2D(channels, kernel_size)
    self.conv_1 = Conv2D(channels, kernel_size)
    self.conv_2 = Conv2D(channels, kernel_size)
    self.conv_3 = Conv2D(channels, kernel_size)

    self.xavier = GlorotUniform()
    noise_scale_shape = (None, img_dim, img_dim, 1)
    self.noise_scale_0 = tf.Variable(
        initializer = xavier(shape=noise_scale_shape))
    self.noise_scale_1 = tf.Variable(
        initializer = xavier(shape=noise_scale_shape))
    self.noise_scale_2 = tf.Variable(
        initializer = xavier(shape=noise_scale_shape))
    self.noise_scale_3 = tf.Variable(
        initializer = xavier(shape=noise_scale_shape))


  def call(self, x, initial=False, training=False):
    # Call layers of network on input x
    # Use the training variable to handle adding layers such as Dropout
    # and Batch Norm only during training
    if !initial:
      x = self.conv_0(x)

    x = x + (self.noise_scale_0 * noise)
    
    y = self.y_0(w)
    y_scale, y_bias = tf.split(y, [self.channels], axis=1)
    # Add instance normalization

    x = self.self.conv_1(x)

    x = x + (self.noise_scale_1 * noise)

    y = self.y_1(w)
    y_scale, y_bias = tf.split(y, [self.channels], axis=1)
    # Add instance normalization

    x = self.conv_2(x)

    x = x + (self.noise_scale_2 * noise)
    
    y = self.y_2(w)
    y_scale, y_bias = tf.split(y, [self.channels], axis=1)
    # Add instance normalization

    x = self.self.conv_3(x)

    x = x + (self.noise_scale_0 * noise)

    y = self.y_3(w)
    y_scale, y_bias = tf.split(y, [self.channels], axis=1)
    # Add instance normalization

    return x
