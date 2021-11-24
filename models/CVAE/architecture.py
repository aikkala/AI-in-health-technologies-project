import tensorflow as tf
from tensorflow.python.ops import math_ops
import numpy as np
from tensorflow import keras as K


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


class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder for 1D signals"""

  def __init__(self, input_dim, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(input_dim,1)),
      #Preprocess(),
      tf.keras.layers.Conv1D(filters=32, kernel_size=6, strides=2, padding='same', activation=None),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Conv1D(filters=64, kernel_size=6, strides=2, padding='same', activation=None),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Conv1D(filters=128, kernel_size=6, strides=2, padding='same', activation=None),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Conv1D(filters=256, kernel_size=6, strides=2, padding='same', activation=None),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(latent_dim*4, activation='leaky_relu'),
      tf.keras.layers.Dense(latent_dim*2)
    ])

    units = 450
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
      tf.keras.layers.Dense(units=units, activation='leaky_relu'),
      tf.keras.layers.Reshape(target_shape=(units,1)),
      tf.keras.layers.Conv1DTranspose(filters=256, kernel_size=3, strides=1, padding='same', activation=None),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=3, strides=1, padding='same', activation=None),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=3, strides=1, padding='same', activation=None),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, strides=1, padding='same', activation=None),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.LeakyReLU(),
      tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=3, padding='same')
    ])

  @tf.function
  def encode(self, x):
    return self.encoder(x)

  @tf.function
  def decode(self, z):
    return self.decoder(z)


  def call(self, x):
    encoded = self.encoder(x)
    z_mean, z_log_sigma = tf.split(encoded, num_or_size_splits=2, axis=1)
    epsilon = tf.random.normal(shape=tf.shape(z_mean), mean=0., stddev=1.)
    z = z_mean + tf.exp(0.5 * z_log_sigma) * epsilon
#    z = self.sample(z_mean, z_log_sigma)
    xr = self.decoder(z)
    return [xr, encoded]

  @staticmethod
  def reconstruction_loss(original, out):
    reconstruction = tf.reduce_mean(tf.square(original - out))
    return reconstruction

  @staticmethod
  def kl_loss(original, cat_tensor):
    z_mean, z_log_sigma = tf.split(cat_tensor, num_or_size_splits=2, axis=1)
    kl = -0.5 * tf.reduce_mean(1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma))
    return 1e-2 * kl

  def sample(self, z_mean, z_log_sigma):
    batch_size = tf.shape(z_mean)[0]
    epsilon = tf.random.normal(shape=(batch_size, self.latent_dim), mean=0., stddev=1.)
    return z_mean + tf.exp(0.5 * z_log_sigma) * epsilon

class MLP(tf.keras.Model):
  """Simple MLP for classification"""

  def __init__(self, input_dim, output_dim):
    super(MLP, self).__init__()
    self.net = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(input_dim,)),
      tf.keras.layers.Dense(128, activation='leaky_relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(128, activation='leaky_relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(output_dim)
    ])

  def call(self, x):
    return self.net(x)