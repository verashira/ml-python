import numpy as np
import os

import load_dataset

feature_names = [
    'area',
    'perimeter',
    'compactness',
    'length of kernel',
    'width of kernel',
    'asymmetry coefficien',
    'length of kernel groove',
]

def load_seeds():
    '''
    feature_names, features, target_names, targets =
        load_seeds()

    Load seeds dataset.

    Returns
    -------
    features_names : Name of every feature
    features : ndarray of training data
    target_names : Name of every available target labels
    target : target values
    '''

    file_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(file_dir, "data", "seeds.tsv")
    data, labels = load_dataset.load_dataset(data_dir)

    target_names = []
    for label in labels:
        if label not in target_names:
            target_names.append(label)
    targets = np.array([target_names.index(l) for l in labels])

    return feature_names, data, target_names, targets
