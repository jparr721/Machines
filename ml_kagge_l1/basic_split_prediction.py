from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

main_file_path = '../data/housing/train.csv'
data = pd.read_csv(main_file_path)

y = data.SalePrice
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

trainX, valX, trainY, valY = train_test_split(X, y, random_state=0)

house_model = DecisionTreeRegressor()
house_model.fit(trainX, trainY)

# Make predictions on the split data that we didn't use to train with
predictions = house_model.predict(valX)
print(mean_absolute_error(valY, predictions))
