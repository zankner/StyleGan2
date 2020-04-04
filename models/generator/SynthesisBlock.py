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

    self.y_0 = Dense(channels)
    self.y_1 = Dense(channels)
    self.y_2 = Dense(channels)
    self.y_3 = Dense(channels)

    self.xavier = GlorotUniform()

    conv_shape = (3, 3, channels, channels)
    self.conv_0 = tf.Variable(initializer = self.xavier(
      shape=conv_shape))
    self.conv_1 = tf.Variable(initializer = self.xavier(
      shape=conv_shape))
    self.conv_2 = tf.Variable(initializer = self.xavier(
      shape=conv_shape))
    self.conv_3 = tf.Variable(initializer = self.xavier(
      shape=conv_shape))

  def build(self, input_shape):
    noise_scale_shape = (input_shape[0], img_dim, img_dim, 1)
    self.noise_scale_0 = self.add_weight(shape=noise_scale_shape,
        initializer='glorot_uniform')
    self.noise_scale_1 = self.add_weight(shape=noise_scale_shape,
        initializer='glorot_uniform')
    self.noise_scale_2 = self.add_weight(shape=noise_scale_shape,
        initializer='glorot_uniform')
    self.noise_scale_3 = self.add_weight(shape=noise_scale_shape,
        initializer='glorot_uniform')


  def call(self, x, initial=False, training=False):
    # Call layers of network on input x
    # Use the training variable to handle adding layers such as Dropout
    # and Batch Norm only during training
    if !initial:
      x = self.upsample_0(x)

    s = self.y_0(w)
    w_prime = self.conv_0 * s
    std = tf.math.reduce_mean(w_prime, self.channels)
    w_prime = w_prime / std

    x = tf.nn.conv2d(x, w_pime)

    x = x + (self.noise_scale_2 * noise)
    
    x = self.upsample_1(x)

    s = self.y_0(w)
    w_prime = self.conv_0 * s
    std = tf.math.reduce_mean(w_prime, self.channels)
    w_prime = w_prime / std

    x = tf.nn.conv2d(x, w_pime)

    x = x + (self.noise_scale_2 * noise)

    return x
