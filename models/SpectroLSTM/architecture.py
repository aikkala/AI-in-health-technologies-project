import tensorflow as tf

class LSTM(tf.keras.Model):
  """Vanilla LSTM"""

  def __init__(self, n_features, n_timesteps, n_hidden, n_output, MASK_VALUE):
    super(LSTM, self).__init__()

    self.MASK_VALUE = MASK_VALUE
    self.lstm = tf.keras.layers.LSTM(n_hidden, input_shape=(n_timesteps, n_features), dropout=0.2,
                                     recurrent_dropout=0.2, return_state=True)
    self.dense1 = tf.keras.layers.Dense(n_hidden, activation='leaky_relu')
    self.dense2 = tf.keras.layers.Dense(n_output)
    self.dropout = tf.keras.layers.Dropout(0.2)

  def call(self, x):

    # Get the mask
    mask = tf.math.reduce_any(x!=self.MASK_VALUE, axis=2)

    # Pass x through the lstm
    lstm_out = self.lstm(x, mask=mask)

    # Also a couple of dense layers
    x = self.dense1(tf.concat(lstm_out, axis=1))
    x = self.dropout(x)

    return self.dense2(x)
