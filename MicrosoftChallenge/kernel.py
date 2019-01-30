import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
INPUT_DIR = '../input/'
# Our data size for test
TEST_CHUNK_SIZE = 10 ** 3
TRAIN_CHUNK_SIZE = 10 ** 4
scaler = StandardScaler()
# Radial basis function kernel since we are doing multi class
support_vector_machine = svm.SVC(kernel='rbf', gamma='scale')


def main():
    for data in pd.read_csv(
            INPUT_DIR + 'train.csv', chunksize=TRAIN_CHUNK_SIZE):
        # Process data at each chunk here
        # new_data = reduce_dimensions(data)
        # process_data(new_data)
        print(data.columns)


def scale_data(data_chunk: pd.DataFrame):
    '''
    Standardizes the data to make it easier to process.

    Parameters
    ----------
    data_chunk: pd.DataFrame - The pandas dataframe
    '''
    scaler.fit(data_chunk)
    return scaler.transform(data_chunk)


def reduce_dimensions(data_chunk: pd.DatFrame):
    '''
    We use principal component analysis to find the
    most reasonable features to use to help make our
    data analysis easier.

    Parameters
    ----------
    data_chunk: pd.DataFrame - The pandas dataframe
    '''
    # First init out PCA object
    pca = PCA(n_components=data_chunk.columns)
    pca.fit(data_chunk)

    # Aquire the principal axes of the data, this helps us see
    # which features are most important
    components = pca.components_
    variance = pca.explained_variance_

    # Vector length shows us the importance
    # TODO (jparr721): Introduce the norm for a more accurate reading
    component_importance = len(components)
    variance_importance = len(variance)


def process_data(data_chunk: pd.DataFrame):
    pass
