import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""Sci-Kit packages vs. the uci wine dataset"""
data = pd.read_csv('https://archive.ics.uci.edu/'
                   'ml/machine-learning-databases/'
                   'wine/wine.data', header=None)

X, y = data.iloc[:, 1:].values, data.iloc[:, 0].values
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit(X_test)
