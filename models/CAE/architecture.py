import tensorflow as tf
from tensorflow.python.ops import math_ops
import numpy as np


class Preprocess(tf.keras.layers.Layer):

  def __init__(self, scale=0.5, offset=1.0, name=None, **kwargs):
    self.scale = scale
    self.offset = offset
    super(Preprocess, self).__init__(name=name, **kwargs)

  def call(self, inputs, training):
    if training:
      s = np.random.uniform(low=1-self.scale, high=1+self.scale)
      o = np.random.uniform(low=-self.offset, high=self.offset)
      inputs = (inputs * s) + o
    return inputs


class CAE(tf.keras.Model):
  """Convolutional autoencoder for 1D signals"""

  def __init__(self, input_dim, latent_dim):
    super(CAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(input_dim,1)),
      Preprocess(),
      tf.keras.layers.Conv1D(filters=32, kernel_size=6, strides=2, padding='same', activation='leaky_relu'),
      tf.keras.layers.Conv1D(filters=64, kernel_size=6, strides=2, padding='same', activation='leaky_relu'),
      tf.keras.layers.Conv1D(filters=128, kernel_size=6, strides=2, padding='same', activation='leaky_relu'),
      tf.keras.layers.Conv1D(filters=256, kernel_size=6, strides=2, padding='same', activation='leaky_relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(latent_dim*2, activation='leaky_relu'),
      tf.keras.layers.Dense(latent_dim)
    ])

    units = 450
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
      tf.keras.layers.Dense(units=units, activation='leaky_relu'),
      tf.keras.layers.Reshape(target_shape=(units,1)),
      tf.keras.layers.Conv1DTranspose(filters=256, kernel_size=3, strides=1, padding='same', activation='leaky_relu'),
      tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=3, strides=1, padding='same', activation='leaky_relu'),
      tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=3, strides=1, padding='same', activation='leaky_relu'),
      tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, strides=1, padding='same', activation='leaky_relu'),
      tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=3, padding='same')
    ])

  @tf.function
  def encode(self, x):
    return self.encoder(x)

  @tf.function
  def decode(self, z):
    return self.decoder(z)


  def call(self, x, only_encode=False):
    z = self.encode(x)
    if only_encode:
      return z
    else:
      return self.decode(z)


class MLP(tf.keras.Model):
  """Simple MLP for classification"""

  def __init__(self, input_dim, output_dim):
    super(MLP, self).__init__()
    self.net = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(input_dim,)),
      tf.keras.layers.Dense(64, activation='leaky_relu'),
      #tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(64, activation='leaky_relu'),
      #tf.keras.layers.Dropout(0.1),
      #tf.keras.layers.Dense(128, activation='leaky_relu'),
      #tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(output_dim)
    ])

  def call(self, x):
    return self.net(x)