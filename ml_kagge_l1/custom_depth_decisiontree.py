from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd


def get_mae(max_leaf_nodes, pred_train, pred_val, target_train, target_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(pred_train, target_train)
    preds_val = model.predict(pred_val)
    mae = mean_absolute_error(target_val, preds_val)

    return mae


def main():
    main_file_path = '../data/housing/train.csv'
    house_data = pd.read_csv(main_file_path)

    y = house_data.SalePrice

    predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

    X = house_data[predictors]

    train_x, val_x, train_y, val_y = train_test_split(X, y, random_state=0)

    leaf_nodes = [5, 50, 500, 5000]

    accurate_nodes = [x for x in range(35, 60, 5)]

    for node in accurate_nodes:
        mae = get_mae(node, train_x, val_x, train_y, val_y)
        print('{} nodes: {}'.format(node, mae))

    for node in leaf_nodes:
        mae = get_mae(node, train_x, val_x, train_y, val_y)
        print('{} nodes: {}'.format(node, mae))

# Best decision tree is 50 nodes deep in this data set
# When refined it appears to be lower... 30-ish range


main()
