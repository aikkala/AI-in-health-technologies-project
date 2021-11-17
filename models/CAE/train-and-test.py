import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as pp
import time
import os
from tqdm import tqdm

from architecture import CAE, MLP

MASK_VALUE = 0
fs1 = 300  # Original sampling rate
fs2 = 50   # Downsampled sampling rate
L = 9      # Length of signal segments

def preprocess(signal):

  # Do the downsampling
  downsampled_indices = np.arange(0, signal.size, fs1/fs2, dtype=int)
  signal = signal[downsampled_indices]

  # Normalise
  signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

  # Cut into overlapping segments
  start_idx = 0
  end_idx = L * fs2
  segments = []
  while end_idx <= len(signal):
    segments.append(signal[int(start_idx):int(end_idx)])
    start_idx += (L / 2) * fs2
    end_idx += (L / 2) * fs2

  return segments

def segment(data, labels):

  preprocessed_data = []
  preprocessed_labels = []

  for sample_idx, signal in enumerate(data):
    segments = preprocess(signal)
    preprocessed_data.extend(segments)
    preprocessed_labels.extend([labels[sample_idx] for _ in range(len(segments))])

  # Convert into a numpy array
  preprocessed_data = np.array(preprocessed_data, dtype=np.float32)
  preprocessed_labels = np.array(preprocessed_labels, dtype=np.float32)

  # Add one singleton dimension because tf expects 3 dimensional tensors for conv1d (batch, dim, 1)
  preprocessed_data = np.reshape(preprocessed_data, preprocessed_data.shape + (1,))

  return preprocessed_data, preprocessed_labels


#def preprocess(data):

  # For each signal:
  #   1) Downsample from 300Hz to 100Hz
  #   2) Cut / pad signal to be 60 seconds long
  #   3) Normalise?

#  fs1 = 300
#  fs2 = 25
#  L = 60
#  preprocessed_data = np.ones((data.shape[0], fs2*L), dtype=np.float32)*MASK_VALUE

#  for sample_idx, signal in enumerate(data):
#    downsampled_indices = np.arange(0, min(signal.size-1, fs1*L), fs1/fs2, dtype=int)
#    signal = signal[downsampled_indices]
#    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
    #signal = signal - min(signal)
    #signal = signal / max(signal)
#    preprocessed_data[sample_idx, :len(signal)] = signal

  # Add one singleton dimension because tf expects 3 dimensional tensors for conv1d (batch, dim, 1)
#  preprocessed_data = np.reshape(preprocessed_data, preprocessed_data.shape + (1,))

#  return preprocessed_data

def classify(data, autoencoder, classifier):

  predicted = np.zeros((data.shape[0]))
  # Go through one signal at a time
  for signal_idx, signal in tqdm(enumerate(data)):

    # Get preprocessed segments
    segments = preprocess(signal)
    segments = np.expand_dims(np.array(segments, dtype=np.float32), 2)

    # Get latent representation
    z = autoencoder.encode(segments)

    # Run through classifier
    logits = classifier.predict(z)
    probs = tf.nn.softmax(logits, axis=1)
    segment_predicted = tf.argmax(probs, axis=1)

    # Use majority voting to determine predicted class, break ties randomly
    labels, counts = np.unique(segment_predicted, return_counts=True)
    predicted[signal_idx] = labels[np.random.choice(np.flatnonzero(counts == counts.max()))]

  return predicted

def map_labels(labels):
  new_labels = np.zeros_like(labels, dtype=int)
  new_labels[labels=="N"] = 0
  new_labels[labels=="O"] = 1
  new_labels[labels=="A"] = 2
  new_labels[labels=="~"] = 3
  return new_labels

def create_datasets(data, labels):

  # Use a validation set to determine when to stop training
  # Seems like easiest way to do this is by using StratifiedKFold
  n_splits = 5
  skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

  # Get indices for train and validation data
  splits = []
  for idx, (_, indices) in enumerate(skf.split(data, labels)):
    splits.append(indices)

  # Extract data
  train_data = data[np.concatenate(splits[:-1])]
  train_labels = map_labels(labels[np.concatenate(splits[:-1])])
  val_data = data[splits[-1]]
  val_labels = map_labels(labels[splits[-1]])

  return train_data, train_labels, val_data, val_labels


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  train_autoencoder = False
  train_classifier = False

  # Load train data
  data = np.load('../../data/sets/train_data.npy', allow_pickle=True)
  labels = np.load('../../data/sets/train_label.npy', allow_pickle=True)

  # Do some preprocessing
  #data = preprocess(data)

  # Create train and validation sets
  batch_size = 64
  train_data, train_labels, val_data, val_labels = create_datasets(data, labels)

  train_data, train_labels = segment(train_data, train_labels)
  val_data, val_labels = segment(val_data, val_labels)

  if train_autoencoder:

    # Initialise model
    autoencoder = CAE(input_dim=train_data.shape[1], latent_dim=64)

    # Train
    es = EarlyStopping(patience=10, verbose=1, min_delta=0.0001, monitor='val_loss', mode='min', restore_best_weights=True)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    history = autoencoder.fit(x=train_data, y=train_data, shuffle=True, batch_size=batch_size, epochs=100, verbose=2,
                    validation_data=(val_data, val_data), callbacks=[es])
    fig, ax = pp.subplots()
    pp.plot(history.history["loss"])
    pp.plot(history.history["val_loss"])
    pp.title('Autoencoder training loss')
    pp.ylabel('MSE')
    pp.xlabel('epochs')
    pp.legend(["train data", "val data"])
    pp.xticks(ticks=history.epoch, labels=[x+1 for x in history.epoch])
    fig.savefig('./autoencoder_loss')
    autoencoder.save('./autoencoder')
  else:
    autoencoder = tf.keras.models.load_model('./autoencoder')

  # Encode training and validation data
  train_z = autoencoder.encode(train_data)
  val_z = autoencoder.encode(val_data)

  # Do classification
  if train_classifier:

    # Initialise model
    classifier = MLP(train_z.shape[1], np.unique(labels).size)

    # Train
    es = EarlyStopping(patience=10, verbose=1, min_delta=0.0001, monitor='val_loss', mode='min', restore_best_weights=True)
    classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics='sparse_categorical_accuracy')
    history = classifier.fit(x=train_z, y=train_labels, shuffle=True, batch_size=batch_size, epochs=100, verbose=2,
                             validation_data=(val_z, val_labels), callbacks=[es])
    fig1, ax1 = pp.subplots()
    pp.plot(history.history["loss"])
    pp.plot(history.history["val_loss"])
    pp.title('Training loss')
    pp.ylabel('categorical crossentropy')
    pp.xlabel('epochs')
    pp.legend(["train data", "val data"])
    pp.xticks(ticks=history.epoch, labels=[x+1 for x in history.epoch])
    fig1.savefig('./classifier_loss')

    fig2, ax2 = pp.subplots()
    pp.plot(history.history["sparse_categorical_accuracy"])
    pp.plot(history.history["val_sparse_categorical_accuracy"])
    pp.title('Classifier training accuracy')
    pp.ylabel('accuracy')
    pp.xlabel('epochs')
    pp.legend(["train data", "val data"])
    pp.xticks(ticks=history.epoch, labels=[x+1 for x in history.epoch])
    fig2.savefig('./classifier_accuracy')

    classifier.save('./classifier')

  else:
    classifier = tf.keras.models.load_model('./classifier')

  predicted = classifier(val_z)
  print("Confusion matrix on validation data")
  print(tf.math.confusion_matrix(val_labels, tf.argmax(predicted, 1)))
  print()


  # Predict on test set
  test_data = np.load('../../data/sets/val1_data.npy', allow_pickle=True)
  test_labels = np.load('../../data/sets/val1_label.npy', allow_pickle=True)
  test_labels = map_labels(test_labels)

  print("Predicting on test set")
  predicted = classify(test_data, autoencoder, classifier)

  print("Confusion matrix on test data")
  print(tf.math.confusion_matrix(test_labels, predicted))
  print()

  from sklearn.metrics import balanced_accuracy_score
  print(f"Balanced accuracy score: {balanced_accuracy_score(test_labels, predicted)}")
