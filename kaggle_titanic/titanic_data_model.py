import numpy as np  # linear algebra
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

data_directory = './data/titanic'


def score_dataset(train_x, test_x, train_y, test_y):
    model = RandomForestRegressor(50)
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    return mean_absolute_error(test_y, predictions)


def get_mae(X, y):
    return -1 * cross_val_score(RandomForestRegressor(50),
                                X, y,
                                scoring='neg_mean_absolute_error').mean()


if __name__ == '__main__':
    imputer = Imputer()
    titanic_raw_train = pd.read_csv('{}/train.csv'.format(data_directory))
    titanic_raw_test = pd.read_csv('{}/test.csv'.format(data_directory))

    titanic_raw_train_new = titanic_raw_train.copy()
    titanic_raw_test_new = titanic_raw_test.copy()

    target = titanic_raw_train.Survived

    cols_with_missing = [col for col in titanic_raw_train.columns
                         if titanic_raw_train[col].isnull().any()]

    # Impute our missing columns
    for col in cols_with_missing:
        titanic_raw_train_new[col] = titanic_raw_train_new[col].isnull()

    titanic_raw_train_new = imputer.fit_transform(titanic_raw_train_new)
    titanic_raw_test_new = imputer.transform(titanic_raw_test_new)

    low_cardinality_cols = [c for c in titanic_raw_train_new.columns if
                            titanic_raw_train_new[c].nunique() < 10 and
                            titanic_raw_train_new[c].dtype == 'object']

    numeric_cols = [c for c in titanic_raw_train_new.columns if
                    titanic_raw_train_new[c].dtype in ['int64', 'float64']]

    titanic_cols = low_cardinality_cols + numeric_cols
    train_predictors = titanic_raw_train_new[titanic_cols]
    test_predictors = titanic_raw_test_new[titanic_cols]

    one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
    one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)

    predictors_without_categoricals = train_predictors.select_dtypes(
            exclude=['object'])

    final_train, final_test = one_hot_encoded_training_predictors.align(
                                one_hot_encoded_test_predictors,
                                join='left',
                                axis=1)

    mae_train_one_hot_encoded = get_mae(final_train, target)
    mae_test_one_hot_encoded = get_mae(final_test, target)

    print('MAE training data: {}'.format(int(mae_test_one_hot_encoded)))
    print('MAE test data: {}'.format(int(mae_test_one_hot_encoded)))
