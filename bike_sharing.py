import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from housing import plot_feature_importances

def load_dataset(filename):

	dataset = pd.read_csv(filename, delimiter = ',')
	X = dataset[dataset.columns[2:13]]
	y = dataset[dataset.columns[-1]]

	feature_names = dataset.columns

	return np.array(X[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names

if __name__=='__main__':

	X, y, feature_names = load_dataset("bike_hour.csv")
	X, y = shuffle(X, y, random_state=7) 

	num_training = int(0.9 * len(X))
	X_train, y_train = X[:num_training], y[:num_training]
	X_test, y_test = X[num_training:], y[num_training:]

	rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10)
	rf_regressor.fit(X_train, y_train)

	y_pred = rf_regressor.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	evs = explained_variance_score(y_test, y_pred) 

	print ("\nRandom Forest regressor performance")
	print ("Mean squared error =", round(mse, 2))
	print ("Explained variance score =", round(evs, 2))

	plot_feature_importances(rf_regressor.feature_importances_, 'Random Forest regressor', feature_names)