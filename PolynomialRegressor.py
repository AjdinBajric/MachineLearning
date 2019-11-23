import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

filename = "data_multivar.txt"

X = []
Y = []

with open(filename, 'r') as f:
	for line in f.readlines():
		a, b, c, d = [float(i) for i in line.split(',')]
		X.append([a, b, c])
		Y.append(d)

num_training = int(len(X) * 0.8)
num_test = len(X) - num_training

X_train = np.array(X[:num_training]).reshape((num_training, 3))
Y_train = np.array(Y[:num_training])

#Test data
X_test = np.array(X[num_training:]).reshape((num_test, 3))
Y_test = np.array(Y[num_training:])

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, Y_train)

polynomial = PolynomialFeatures(degree = 10)

#Represent the datapoints in terms of the coefficients of the polynomial
#Formula: [1, a, b, c, a^2, ab, ac, b^2, bc, c^2]

X_train_transformed = polynomial.fit_transform(X_train)

datapoint = np.array(X_train[0]).reshape(1,3)
print("\nDatapoint = ", datapoint)

poly_datapoint = polynomial.fit_transform(datapoint)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, Y_train)

print("\nLinear regression:", linear_regressor.predict(datapoint)[0])
print("\nPolynomial regression:", poly_linear_model.predict(poly_datapoint)[0])
print("\nActual value: ", Y[0])
