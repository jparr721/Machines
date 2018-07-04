from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd


main_file_path = '../data/housing/train.csv'
data = pd.read_csv(main_file_path)

y = data.SalePrice

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

X = data[predictors]

house_data = DecisionTreeRegressor()

house_data.fit(X, y)

# In-Sample score (innacurate)
predicted_house_price = house_data.predict(X)
mean_absolute_error(y, predicted_house_price)
