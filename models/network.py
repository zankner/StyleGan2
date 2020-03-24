import tensorflow as tf
# Import layers:
from tensorflow.keras.layers import (
    Dense, Flatten, Conv2D, Conv2DTranspose, 
    BatchNormalization, concatenate, MaxPool2D
)
from tensorflow.keras import Model

#Defining network Below:
class Network(Model):
  def __init__(self):
    super(UNet, self).__init__()
    # Define layers of the network:


  def call(self, x, training=False):
    # Call layers of network on input x
    # Use the training variable to handle adding layers such as Dropout
    # and Batch Norm only during training

    return x
