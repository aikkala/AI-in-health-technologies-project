import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as pp
from scipy.signal import spectrogram
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
from sklearn.metrics import balanced_accuracy_score

from architecture import LSTM

MASK_VALUE = -100

def preprocess(data):

  # Original sampling rate
  fs = 300

  # Length of each segment
  nperseg = 4*fs

  # Overlapping segments
  noverlap = nperseg//2

  # Cutoff signal length at 60 seconds
  max_len = 60*fs

  nfft = 2048

  # Find spectrogram shape first
  f, _, S = spectrogram(np.zeros((max_len,)), fs=fs, nperseg=nperseg, noverlap=noverlap, detrend='linear', nfft=nfft)

  # Cutoff frequencies at 10 Hz
  fs_cutoff = 10
  idx_cutoff = np.where(f > fs_cutoff)[0][0]

  # Pad spectrograms so easier to work with
  spectrograms = np.ones((data.shape[0], S.shape[1], idx_cutoff)) * MASK_VALUE
  for signal_idx, signal in enumerate(data):

    # Do the spectrogram
    f, t, S = spectrogram(signal[:max_len], fs=fs, nperseg=nperseg, noverlap=noverlap, detrend='linear', nfft=nfft)

    # Discard everything below 15Hz and take the logarithm
    S = np.log(S[:idx_cutoff])

    # Normalise?
    S = S - np.mean(S) / (np.std(S) + 1e-7)

    spectrograms[signal_idx, :S.shape[1]] = S.T

  return spectrograms

def classify(data, autoencoder, classifier):

  predicted = np.zeros((data.shape[0]))
  # Go through one signal at a time
  for signal_idx, signal in tqdm(enumerate(data)):

    # Get preprocessed segments
    segments = preprocess(signal)
    segments = np.expand_dims(np.array(segments, dtype=np.float32), 2)

    # Get latent representation
    enc = autoencoder.encoder(segments)

    # Run through classifier
    logits = classifier.predict(enc)
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
  train = False

  # Load train data
  data = np.load('../../data/sets/train_data.npy', allow_pickle=True)
  labels = np.load('../../data/sets/train_label.npy', allow_pickle=True)

  # Add val1 data to train data
  val1_data = np.load('../../data/sets/val1_data.npy', allow_pickle=True)
  val1_labels = np.load('../../data/sets/val1_label.npy', allow_pickle=True)
  data = np.concatenate([data, val1_data])
  labels = np.concatenate([labels, val1_labels])

  # Add val2 data to train data
  val2_data = np.load('../../data/sets/val2_data.npy', allow_pickle=True)
  val2_labels = np.load('../../data/sets/val2_label.npy', allow_pickle=True)
  data = np.concatenate([data, val2_data])
  labels = np.concatenate([labels, val2_labels])

  # Set seed so we don't create new train/validation split each time
  np.random.seed(123)

  # Create train and validation sets
  batch_size = 64
  train_data, train_labels, val_data, val_labels = create_datasets(data, labels)

  train_data = preprocess(train_data)
  val_data = preprocess(val_data)

  if train:

    # Initialise model
    model = LSTM(n_timesteps=train_data.shape[1], n_features=train_data.shape[2], n_hidden=256, n_output=4,
                 MASK_VALUE=MASK_VALUE)

    # Use class weights to minimize effect of imbalanced data sets
    num_labels = {l: sum(train_labels == l) for l in np.unique(train_labels)}
    max_label = max(num_labels.values())
    class_weight = {l: max_label / num_labels[l] for l in num_labels}

    # Train
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
    es = EarlyStopping(patience=20, verbose=1, min_delta=0.0001, monitor='val_loss', mode='min', restore_best_weights=True)
    history = model.fit(x=train_data, y=train_labels, shuffle=True, batch_size=batch_size, epochs=100, verbose=2,
                        validation_data=(val_data, val_labels), callbacks=[es], class_weight=class_weight)

    fig, ax = pp.subplots()
    pp.plot(history.history["loss"])
    pp.plot(history.history["val_loss"])
    pp.title('Training loss')
    pp.ylabel('Categorical Crossentropy')
    pp.xlabel('epochs')
    pp.legend(["train data", "val data"])
    #pp.xticks(ticks=history.epoch, labels=[x+1 for x in history.epoch])
    fig.savefig('./training_loss')

    fig2, ax2 = pp.subplots()
    pp.plot(history.history["sparse_categorical_accuracy"])
    pp.plot(history.history["val_sparse_categorical_accuracy"])
    pp.title('Training accuracy')
    pp.ylabel('accuracy')
    pp.xlabel('epochs')
    pp.legend(["train data", "val data"])
    #pp.xticks(ticks=history.epoch, labels=[x+1 for x in history.epoch])
    fig2.savefig('./training_accuracy')

    model.save('./lstm')
  else:
    model = tf.keras.models.load_model('./lstm')


  predicted = tf.argmax(model.predict(val_data), 1)
  print("Confusion matrix on validation data")
  print(tf.math.confusion_matrix(val_labels, predicted))
  print()
  print(f"Balanced accuracy score on validation data: {balanced_accuracy_score(val_labels, predicted)}")
  print()

  # Predict on test set
  test_data = np.load('../../data/sets/val3_data.npy', allow_pickle=True)
  test_labels = np.load('../../data/sets/val3_label.npy', allow_pickle=True)
  test_data = preprocess(test_data)
  test_labels = map_labels(test_labels)

  print("Predicting on test set")
  predicted = tf.argmax(model.predict(test_data), 1)

  print("Confusion matrix on test data")
  print(tf.math.confusion_matrix(test_labels, predicted))
  print()

  print(f"Balanced accuracy score on test data: {balanced_accuracy_score(test_labels, predicted)}")
