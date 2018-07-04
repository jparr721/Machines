from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    main_file_path = '../data/housing/train.csv'
    house_data = pd.read_csv(main_file_path)

    y = house_data.SalePrice
    predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

    X = house_data[predictors]

    train_x, val_x, train_y, val_y = train_test_split(X, y, random_state=0)

    forest = RandomForestRegressor()
    forest.fit(train_x, train_y)

    house_predictions = forest.predict(val_x)
    print(mean_absolute_error(val_y, house_predictions))


main()
