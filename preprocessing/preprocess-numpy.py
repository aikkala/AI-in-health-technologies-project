import numpy as np
import os
import

def parse_labels(reference_file):

  # Read the csv file



if __name__ == "__main__":

  # Define some paths
  raw_data_folder = '../data/raw'
  preprocessed_data_foldewr = '../data/preprocessed'

  # Load train and validation sets
  train_data = np.load(os.path.join(raw_data_folder, "raw_train_data.npy"))
  val_data = np.load(os.path.join(raw_data_folder, "raw_validation_data.npy"))

  # Load labels for train and validation sets
  train_labels = parse_labels(os.path.join(raw_data_folder, 'training2017/REFERENCE.CSV'))
  val_labels = parse_labels(os.path.join(raw_data_folder, 'validation/REFERENCE.CSV'))

  # What kind of pre-processing do we need here? Normalise?