# This is Arjun Koshal's Final CS Project for CS 110
# I pledge my honor that I have abided by the Stevens Honor System
# For this project, I wish to demonstrate the functionality of machine learning
# in Python. I am going to allow the user to input a CSV file and have the
# program examine the file and create the line of best fit. It will also
# be able to determine the accuracy of the line, identify the coefficients and
# intercepts, and predict the future statistics.

# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as mat



# Read file

file_input = str(input("Please enter a CSV file: "))
if ".csv" not in file_input:
    file_input += ".csv"
data = pd.read_csv(file_input)
data.head()

# Input for x and y axis

x_axis = str(input("\nPlease enter the x-axis: "))
y_axis = str(input("Please enter the y-axis: "))

data = data[[x_axis, y_axis]]

# x axis vs y axis:

mat.scatter(data[x_axis], data[y_axis], color="purple")
mat.xlabel(x_axis)
mat.ylabel(y_axis)


# 1. Linear Regression

# Training and testing data using 80:20:

train = data[:(int((len(data)*0.8)))]
test = data[(int((len(data)*0.2))):]

# Model the data:

from sklearn import linear_model
regression = linear_model.LinearRegression()

train_x = np.array(train[[x_axis]])
train_y = np.array(train[[y_axis]])
regression.fit(train_x, train_y)

# The coefficient and intercept:

print("\nCoefficient: %.2f" % regression.coef_)
print("Intercept: %.2f" % regression.intercept_)

# The plot for data:

mat.scatter(train_x, train_y, color="purple")
mat.plot(train_x, regression.coef_*train_x + regression.intercept_, "-r")
mat.xlabel(x_axis)
mat.ylabel(y_axis)
mat.show()

# Prediction for values:

def get_predictions(m, b, x):
    y = m * x + b
    return y

# Predicting dependent value for the future:

future_pred = float(input("\nPlease enter the future prediction value you want to calculate: "))
estimate = get_predictions(future_pred, regression.intercept_[0], regression.coef_[0][0])
print("\nWhen", x_axis, "is", future_pred, "then", y_axis, "is %.2f" % estimate)

# Checking various accuracy:

from sklearn.metrics import r2_score

future_x = np.array(test[[x_axis]])
future_y = np.array(test[[y_axis]])
predict = regression.predict(future_x)

print("\nMean absolute error: %.2f" % np.mean(np.absolute(predict - future_y)))
print("Mean sum of squares (MSE): %.2f" % np.mean((predict - future_y) ** 2))
print("R^2 score: %.2f" % r2_score(predict, future_y))

