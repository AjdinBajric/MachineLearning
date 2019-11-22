import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.metrics as sm
import _pickle as pickle
from sklearn.preprocessing import PolynomialFeatures

filename = "data_singlevar.txt"
X = []
Y = []

with open(filename, 'r') as f:
	for line in f.readlines():
		xt, yt = [float(i) for i in line.split(',')]
		X.append(xt)
		Y.append(yt)

#Divide Dataset into training and testing dataset
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

#Training data
X_train = np.array(X[:num_training]).reshape((num_training, 1))
Y_train = np.array(Y[:num_training])

#Test data
X_test = np.array(X[num_training:]).reshape((num_test, 1))
Y_test = np.array(Y[num_training:])

#Training a model
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, Y_train)

Y_train_pred = linear_regressor.predict(X_train)
plt.figure()
plt.scatter(X_train, Y_train, color = 'green')

plt.plot(X_train, Y_train_pred, color = 'red', linewidth = 2)
plt.title("Training Data")
plt.show()

y_test_pred = linear_regressor.predict(X_test)

plt.scatter(X_test, Y_test, color='green')
plt.plot(X_test, y_test_pred, color='red', linewidth=2)
plt.title('Test data')
plt.show()

#Computing regression accuracy
print("\nMean absolute error = ", round(sm.mean_absolute_error(Y_test, y_test_pred),2))
print("\nMean squared error = ", round(sm.mean_squared_error(Y_test, y_test_pred),2))
print("\nMedian absolute error = ", round(sm.median_absolute_error(Y_test, y_test_pred),2))
print("\nExplained variance score = ", round(sm.explained_variance_score(Y_test, y_test_pred),2))
print("\nR2 score = ", round(sm.r2_score(Y_test, y_test_pred),2))

#Saving a model 
output_model_file = 'saved_model.pkl'

with open(output_model_file, 'wb') as f:
	pickle.dump(linear_regressor, f)

#Load a model
with open(output_model_file, 'rb') as f:
	model_linregr = pickle.load(f)

#Here, we just loaded the regressor from the file into the model_linregr variable.
#You can compare the preceding result with the earlier result to confirm that it's
#the same.
y_test_pred_new = model_linregr.predict(X_test)
print("\nNew mean absolute error = ", round(sm.mean_absolute_error(Y_test, y_test_pred_new),2))

#Ridge Regressor
# As alpha gets closer to 0, the ridge
#regressor tends to become more like a linear regressor with ordinary least squares.
#So, if you want to make it robust against outliers, you need to assign a higher value
#to alpha.
ridge_regressor = linear_model.Ridge(alpha = 0.01, fit_intercept = True, max_iter = 10000)

ridge_regressor.fit(X_train, Y_train)
y_test_pred_ridge = ridge_regressor.predict(X_test)
print("\nMean absolute error = ", round(sm.mean_absolute_error(Y_test, y_test_pred_ridge),2))
print("\nMean squared error = ", round(sm.mean_squared_error(Y_test, y_test_pred_ridge),2))
print("\nMedian absolute error = ", round(sm.median_absolute_error(Y_test, y_test_pred_ridge),2))
print("\nExplained variance score = ", round(sm.explained_variance_score(Y_test, y_test_pred_ridge),2))
print("\nR2 score = ", round(sm.r2_score(Y_test, y_test_pred_ridge),2))

#Building a polynomial regressor

polynomial = PolynomialFeatures(degree = 3)

#Represent the datapoints in terms of the coefficients of the polynomial
X_train_transformed = polynomial.fit_transform(X_train)


print(X_train_transformed)
print(X_train)