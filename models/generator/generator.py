import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, Conv2DTranspose, 
    BatchNormalization, concatenate, MaxPool2D
)
from tensorflow.keras import Model

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
    
    synthesis_network = []
    for block_dim in range(18):
      synthesis_network.append(
          synth_block(2**(block_dim)))


  def call(self, x, training=False):
    # Call layers of network on input x
    # Use the training variable to handle adding layers such as Dropout
    # and Batch Norm only during training

    return x
