"""
A script to process and clean forest census data so that it is prepared
to be run through machine learning models.
 
"""

import numpy as np
import scipy as sp
import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce


def _load_data(file_path, quads):
    """
    Returns the "raw" data with the desired number of quadrats present.

    Parameters
    ----------
    file_path: string with the filepath to a census dataset
    quads: number of quadrats to consider
    """
    # load all data
    raw_data = pd.read_csv(file_path)
    return raw_data.loc[raw_data['quadrat'] <= quads]


def clean_t1(ds):
    # keep only IDs, species, xy coords, and dbh
    ds = ds[['treeID', 'sp', 'gx', 'gy', 'dbh']]
    ds = ds.rename(columns={"dbh": "dbh1", "treeID": "treeID1"})


def clean_t2(ds):
    # keep only the IDs and the expected dbhs
    expected_labels = ds[['treeID', 'dbh']]
    expected_labels = expected_labels.rename(
        columns={"dbh": "dbh2", "treeID": "treeID2"})


def encode_drop_null(t1, t2):
    """
    returns numpy arrays of the two times - (t1, t2)
    """
    t1 = clean_t1(t1)
    t2 = clean_t2(t2)

    # encode the species into categorical data rather than strings
    encoder = ce.BinaryEncoder(cols=['sp'], return_df=True)
    t1 = encoder.fit_transform(t1)
    ds_combined = pd.concat([t1, t2], axis=1)

    # drop rows with any NaN values in either t1 or t2 dbhs
    ds_combined_clean = ds_combined.dropna()

    # convert these non-null datasets into numpy arrays and return
    t1_updated = ds_combined_clean[t1.columns].to_numpy()
    t2_updated = ds_combined_clean[t2.columns].to_numpy()
    return (t1_updated, t2_updated)


def growth_labels(t1, t2):
    # change the labels so that they show GROWTH, not future DBH - better for readability of loss
    t2[:, 1] = t2[:, 1] - t1[:, 10]
    return np.where(t2 < 0, 0, t2)


def split(t1, t2, split_prop):
    X_train, X_test, y_train, y_test = train_test_split(t1, t2, test_size=0.3)
    feats = X_train[:, 1:]
    labels = y_train[:, 1]

    test_ids = X_test[:, 0]
    test_feats = X_test[:, 1:]
    return [feats, labels, test_ids, test_feats]


def clean(file_paths, num_quads=4924, split_prop=0.3):
    """
    Cleans the data and returns the training and testing splits required.

    Returns a list of form [feats, labels, test_feats, test_ids].

    With a machine learning model ABC, the general flow will work like:
    ABC = ABCRegressor()
    ABC.fit(feats, labels)
    preds = ABC.predict(test_feats)
    # line below is just for ease of user interpretation
    preds_matrix = np.column_stack((test_ids, preds))
    error = measure_of_error(y_test[:,1], preds)

    Parameters
    ----------
    file_paths: a tuple of two filepaths, the first being that of the initial 
      dataset and the second the path of the later dataset
    num_quads: the number of quadrats to consider (default: 4924, the entire
      dataset when looking at BCI specifically)
    split: the proportion of the data that should be used for testing (default 
      value is 0.3, so 30% will be used on testing and 70% will be used on
       training)
    """
    data_t1 = _load_data(file_paths[0])
    data_t2 = _load_data(file_paths[1])
    data_t1, data_t2 = encode_drop_null(data_t1, data_t2)
    data_t2 = growth_labels(data_t1, data_t2)
    return split(data_t1, data_t2, split_prop)
