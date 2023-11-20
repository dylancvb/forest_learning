"""
A script to process and clean forest census data so that it is prepared
to be run through machine learning models.
 
"""
import numpy as np
import scipy as sp
import pandas as pd
import category_encoders as ce


# def preprocess(file_path, quads):
#     raw_data = pd.read_csv(file_path)
#     raw_data =  raw_data.loc[raw_data['quadrat'] <= quads]
#     return 1

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
    return ds

    print("t1 is"+str(type(ds)))


def clean_t2(ds):
    # keep only the IDs and the expected dbhs
    expected_labels = ds[['treeID', 'dbh']]
    expected_labels = expected_labels.rename(
        columns={"dbh": "dbh2", "treeID": "treeID2"})
    return ds
    print("t2 is"+str(type(ds)))


def encode_drop_null(t1: pd.DataFrame, t2: pd.DataFrame):
    """
    returns numpy arrays of the two times - (t1, t2)
    """

    t1_clean = clean_t1(t1)
    t2_clean = clean_t2(t2)
    # print("t1 is"+str(type(t1_clean)))
    # print("t2 is"+str(type(t2_clean)))

    # encode the species into categorical data rather than strings
    encoder = ce.OneHotEncoder(cols=['sp'], return_df=True)

    t1_clean = encoder.fit_transform(t1_clean)

    ds_combined = pd.concat([t1_clean, t2_clean], axis=1)
    print(type(ds_combined))

    # drop rows with any NaN values in either t1 or t2 dbhs
    ds_combined_clean = pd.DataFrame.dropna(ds_combined)

    print("ds combined clean is" + str(ds_combined_clean))

    # convert these non-null datasets into numpy arrays and return
    t1_updated = ds_combined_clean[t1_clean.columns].to_numpy()
    t2_updated = ds_combined_clean[t2_clean.columns].to_numpy()
    # print(t1_updated)
    # print(t2_updated)
    return (t1_updated, t2_updated)


def growth_labels(t1, t2):
    # change the labels so that they show GROWTH, not future DBH - better for readability of loss
    t2[:, 1] = t2[:, 1] - t1[:, 10]
    return np.where(t2 < 0, 0, t2)


def split(t1, t2, split_prop):
    print(t1)
    X_train, X_test, y_train, y_test = train_test_split(
        t1, t2, test_size=split_prop)
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
    data_t1 = _load_data(file_paths[0], num_quads)
    print("data-t1 is"+str(type(data_t1)))
    data_t2 = _load_data(file_paths[1], num_quads)
    print("data-t2 is"+str(type(data_t1)))
    data_t1, data_t2 = encode_drop_null(data_t1, data_t2)
    data_t2 = growth_labels(data_t1, data_t2)
    # don't return this type of split, just return the cleaned x and y data!!

    # can do this in another function.

    # return split(data_t1, data_t2, split_prop)


# another function should be making distance matrices

# another function should be the permutations of this

# another fn should be framework for printing and creating graphs!!


# load all data
def returndataframe(t1filepath, t2filepath, quadrats=-1):
    t1 = pd.read_csv(t1filepath)
    t2 = pd.read_csv(t2filepath)

    if quadrats > 0:
        t1 = t1.loc[t1['quadrat'] <= quadrats]
        t2 = t2.loc[t2['quadrat'] <= quadrats]

    featurest1 = t1[['treeID', 'sp', 'dbh', 'gx', 'gy']]
    featurest1 = featurest1.rename(
        columns={"dbh": "dbh1", "treeID": "treeID1"})

    expected_labels = t2[['treeID', 'dbh']]
    expected_labels = expected_labels.rename(
        columns={"dbh": "dbh2", "treeID": "treeID2"})

    # referred to as a in ensemble_model
    sp_num = featurest1.sp.unique().shape[0]

    encoder = ce.OneHotEncoder(cols=['sp'], return_df=True)
    featurest1 = encoder.fit_transform(featurest1)

    df_combined = pd.concat([featurest1, expected_labels], axis=1)
    df_combined_clean = df_combined.dropna()

    # right now, just eliminate all data points where expected labels are less than 0
    df_combined_clean['dbh2'] = df_combined_clean['dbh2'] - \
        df_combined_clean['dbh1']

    df_combined_clean = df_combined_clean[df_combined_clean.dbh2 >= 0]

    # split back up and convert the dataframes to numpy arrays
    df_clean = df_combined_clean[featurest1.columns]
    labels_clean = df_combined_clean[expected_labels.columns]

    featurest1 = df_clean.to_numpy()
    expected_labels = labels_clean.to_numpy()

    featurest1 = featurest1.astype(np.float32)
    expected_labels = expected_labels.astype(np.float32)

    return featurest1, expected_labels, sp_num


def get_neighborhood(feats, sp_num, num_neighbors):
    ids = feats[:, 0]
    # X WOULD BE a +1, y would be a+2
    x_coordinates = feats[:, sp_num+2]
    y_coordinates = feats[:, sp_num+3]
    coord_matrix = np.column_stack((x_coordinates, y_coordinates))
    spatial_tree = sp.spatial.KDTree(coord_matrix)

    nn_dist_matrix2 = np.zeros((len(coord_matrix), num_neighbors+1))
    nn_ind_matrix2 = np.zeros((len(coord_matrix), num_neighbors+1))
    nn_feats = feats[:, 0:sp_num+2]
    feats_matrix = np.zeros(
        (len(coord_matrix), ((num_neighbors+1)*(sp_num+2))))

    for i, tree in enumerate(coord_matrix):
        dist, ind = spatial_tree.query(tree, k=num_neighbors+1)
        nn_ind_matrix2[i] = ids[ind]
        nn_dist_matrix2[i] = dist

        nn_row = nn_feats[i].reshape(1, sp_num+2)
        inc = 0
        for j in nn_ind_matrix2[i][1:]:
            row_ind = np.where(feats[:, 0] == j)
            real_row = (feats[row_ind])
            distance = dist[1:][inc].reshape(1, 1)
            dbh = real_row[:, sp_num+1].reshape(1, 1)
            nn_row = np.hstack(
                (nn_row, distance, real_row[:, 1:sp_num+1], dbh))
            inc += 1

        feats_matrix[i] = nn_row

    return feats_matrix


def neighborhood_naive(feats, sp_num):
    # check if tree species = tree species of neighbor. if yes, 1. if no, 0.
    feats[:, 1:(sp_num+1)]


def preparedata(feats, labels, sp_num, whichlevel: int):

    labels = labels[:, 1]

    if whichlevel == 1:
        # just DBH, no species information at all

        feats = feats[:, (sp_num+1)]

    if whichlevel == 2:
        feats = feats[:, 1:]
        # DBH and species information

    if whichlevel == 3:
        # full neighborhood information
        num_neighbors = 20
        feats = get_neighborhood(feats, sp_num, num_neighbors)
        return

    if whichlevel == 4:

        # neighborhood with only "same" or "different" species
        return

    return feats, labels
