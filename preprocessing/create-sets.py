import numpy as np
import os
from sklearn.model_selection import StratifiedKFold

def parse_labels(reference_file):
  # Return parsed data in form np.array((subject_name, label), dtype=object)
  return np.genfromtxt(reference_file, delimiter=',', dtype=str)


def save_set(output_data_folder, prefix, data, labels, indices):
  data_file = os.path.join(output_data_folder, prefix+"_data.npy")
  label_file = os.path.join(output_data_folder, prefix+"_label.npy")
  np.save(data_file, data[indices])
  np.save(label_file, labels[indices])

if __name__ == "__main__":

  # Define some paths
  raw_data_folder = '../data/raw'
  output_data_folder = '../data/sets'

  # Load train and validation sets
  train_data = np.load(os.path.join(raw_data_folder, "raw_train_data.npy"), allow_pickle=True)
  val_data = np.load(os.path.join(raw_data_folder, "raw_validation_data.npy"), allow_pickle=True)

  # Load labels for train and validation sets
  train_labels = parse_labels(os.path.join(raw_data_folder, 'training2017/REFERENCE.csv'))
  val_labels = parse_labels(os.path.join(raw_data_folder, 'validation/REFERENCE.csv'))

  # Make sure data and labels are in the same order
  assert np.array_equal(train_data[:, 0], train_labels[:, 0]), "Train data not in correct order"
  assert np.array_equal(val_data[:, 0], val_labels[:, 0]), "Validation data not in correct order"

  # Concatenate data, and do splits
  data = np.concatenate([train_data[:, 1], val_data[:, 1]])
  labels = np.concatenate([train_labels[:, 1], val_labels[:, 1]])

  print(f"In total we have {labels.shape[0]} samples:"
        f"  normals: {sum(labels=='N')}"
        f"  AF: {sum(labels=='A')}"
        f"  others: {sum(labels=='O')}"
        f"  noisy: {sum(labels=='~')}")

  # Must take stratification into account since class numbers are not balanced. Divide data into 10 folds,
  # then from training set using five of those, test set using two of them, and then three sets are left for
  # validation. Hence, the data is split into 50% training, 20% testing, and three 10% folds for validation
  n_splits = 10
  #train_size = 5
  #test_size = 2
  skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

  # Need to loop through the splits since they are yielded
  splits = []
  for idx, (_, indices) in enumerate(skf.split(data, labels)):
    splits.append(indices)
    print(f"Set {idx} | normals: {sum(labels[indices]=='N')}, AF: {sum(labels[indices]=='A')}, others: {sum(labels[indices]=='O')}, noisy: {sum(labels[indices]=='~')}")

  # Save train set
  save_set(output_data_folder, "train", data, labels, np.concatenate(splits[:5]))

  # Form test set
  save_set(output_data_folder, "test", data, labels, np.concatenate(splits[5:7]))

  # Form validation sets
  for i in range(3):
    save_set(output_data_folder, f"val{i+1}", data, labels, splits[7+i])
