import numpy as np
import os
import scipy.io

def collect_data(matlab_data_folder):

  # Get mat files
  files = [os.path.join(matlab_data_folder, f) for f in os.listdir(matlab_data_folder) if f.endswith('.mat')]
  files = np.sort(files)

  # Collect all data into a numpy array (note that signals can be of different lengths)
  data = np.empty((len(files),), dtype=object)

  # Go through all data files, collect into one array
  for idx, file in enumerate(files):
    mat = scipy.io.loadmat(file)
    data[idx] = mat['val'].squeeze()

  return data

if __name__ == "__main__":

  # Location of original matlab data
  raw_data_folder = '../data/raw'

  # Get training data
  matlab_train_data_folder = os.path.join(raw_data_folder, 'training2017')
  train_data = collect_data(matlab_train_data_folder)

  # Save training data
  np.save(os.path.join(raw_data_folder, 'raw_train_data.npy'), train_data)

  # Get test data
  matlab_test_data_folder = os.path.join(raw_data_folder, 'validation')
  test_data = collect_data(matlab_test_data_folder)

  # Save test data
  np.save(os.path.join(raw_data_folder, 'raw_validation_data.npy'), test_data)