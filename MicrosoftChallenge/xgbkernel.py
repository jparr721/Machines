import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
from xgboost import XGBRegressor

INPUT_DIR = '../input/'
scaler = StandardScaler()


def train():
    data = pd.read_csv(INPUT_DIR + 'train.csv')
    y = data['HasDetections']
    X = data.drop(['HasDetections'])

def quick_train():
    xgbmodel = XGBRegressor()
    
    for data in pd.read_csv(INPUT_DIR + 'train.csv', chunksize=10):
        y = data['HasDetections']
        X = data.drop(['HasDetections'], axis=1)
        
        # Get numerical columns
        X = X._get_numeric_data()
        
        # Drop NaNs
        X.dropna(inplace=True, axis=1)

        print(X.head)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, random_state=0)
        
        xgbmodel.fit(X_train, y_train, verbose=False)
        predictions = xgbmodel.predict(X_test)
        
        print('MAE: {}'.format(str(mean_absolute_error(predictions, y_test))))


if __name__=='__main__':
    quick_train()
