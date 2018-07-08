import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

cityPop = []
profit = []

# read the data
with open('../data/ex1/ex1data1.txt') as truckinfo:
    for truck in truckinfo:
        current = truck.split(',')
        cityPop.append(current[0].rstrip())
        profit.append(current[1].rstrip())

population = [float(x) for x in cityPop]
profits = [float(x) for x in profit]

# train our linear model on the data -- This is useful for predictions etc.
# data_reg = linear_model.LinearRegression()
# data_reg.fit(population, profits)


# scatter plot
plt.scatter(population, profits)

# get our trend line
z = np.polyfit(population, profits, 1)
p = np.poly1d(z)

# plot our new data
# plot.plot(population, p(data_reg.predict(population)), 'r--') -- This is useful for prediction
plt.plot(population, p(population), 'r--')
plt.show()
