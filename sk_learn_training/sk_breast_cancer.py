import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


"""Sci-Kit Learn applied to the breast cancer uci data"""
data = pd.read_csv('http://archive.ics.uci.edu/ml/'
                   'machine-learning-databases'
                   '/breast-cancer-wisconsin/wdbc.data',
                   header=None)

# Encode the class labels so they can be read by our algorithms
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_

X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                     test_size=0.20,
                     stratify=y,
                     random_state=1)


def sk_pipeline():
    pipe_lr = make_pipeline(StandardScaler(),
                            PCA(n_components=2),
                            LogisticRegression(random_state=1))

    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    print('test accuracy: {}'.format(pipe_lr.score(X_test, y_test)))


def sk_k_fold_pipeline():
    pipe_lr = make_pipeline(StandardScaler(),
                            PCA(n_components=2),
                            LogisticRegression(random_state=1))
    kfold = StratifiedKFold(n_splits=10,
                            random_state=1).split(X_train, y_train)
    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)
        print('Fold: {}, Class dist: {}, Acc: {}'
              .format(k+1, np.bincount(y_train[train]), score))
    print('Mean accuracy: {} +/- {}'.format(np.mean(scores), np.std(scores)))

    # A little bit faster now...
    scores = cross_val_score(estimator=pipe_lr,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             n_jobs=-1)
    print('CV Accuracy Scores: {}'.format(scores))
    print('CV accuracy: {} +/- {}'.format(np.mean(scores), np.std(scores)))


# sk_pipeline()
sk_k_fold_pipeline()
