import pandas as pd

main_file_path = '../data/housing/test.csv'
data = pd.read_csv(main_file_path)

columns = ['Neighborhood', 'LandSlope']

house_condition = data[columns]

print(house_condition.describe())
