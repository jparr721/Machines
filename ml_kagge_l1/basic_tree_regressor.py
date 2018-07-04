from sklearn.tree import DecisionTreeRegressor
import pandas as pd

main_file_path = '../data/housing/train.csv'

house_data_raw = pd.read_csv(main_file_path)

# our prediction target
y = house_data_raw.SalePrice

# our predictors
predictors = [
        'LotArea',
        'YearBuilt',
        '1stFlrSF',
        '2ndFlrSF',
        'FullBath',
        'BedroomAbvGr',
        'TotRmsAbvGrd'
        ]

X = house_data_raw[predictors]

house_data = DecisionTreeRegressor()

house_data.fit(X, y)

print("Predictions for the first 5 houses:")
print(X.head())
print("Predictions:")
print(house_data.predict(X.head()))
