import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
INPUT_DIR = '../input/'
# Our data size for test
TEST_CHUNK_SIZE = 10 ** 3
TRAIN_CHUNK_SIZE = 10 ** 4
scaler = StandardScaler()
# Radial basis function kernel since we are doing multi class
support_vector_machine = svm.SVC(kernel='rbf', gamma='scale', decision_function_shape='ovo')
    
def compute_average_score(scores: list):
    total = 0
    [total += i for i in scores]
    return total/len(scores)
    

def scale_data(data_chunk: pd.DataFrame):
    '''
    Standardizes the data to make it easier to process.

    Parameters
    ----------
    data_chunk: pd.DataFrame - The pandas dataframe
    '''
    scaler.fit(data_chunk)
    return scaler.transform(data_chunk)


def reduce_dimensions(X: pd.DatFrame, y: pd.DataFrame):
    '''
    We use principal component analysis to find the
    most reasonable features to use to help make our
    data analysis easier.

    Parameters
    ----------
    data_chunk: pd.DataFrame - The pandas dataframe
    '''
    # split our data for the classifier to work with
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    
    # Standardize our data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(.95)
    pca.fit(X_train)
    
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def analyze(classifier: any,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_train: pd.DataFrame,
            y_test: pd.DataFrame):
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    return score
    

def main():
    scores = []
    for data in pd.read_csv(
            INPUT_DIR + 'train.csv', chunksize=TRAIN_CHUNK_SIZE):
        X = data[:len(data.columns)-1]
        y = data[len(data.columns)-1:]
        
        # Process data at each chunk here
        # First reduce dimensions
        X_train, X_test, y_train, y_test = reduce_dimensions(X, y)
        score = analyze(
            support_vector_machine, X_train, X_test, y_train, y_test)
        print(score)
        scores.append(score)
    
    average_score = compute_average_score(scores)
    return average_score


if __name__=='__main__':
    main()
