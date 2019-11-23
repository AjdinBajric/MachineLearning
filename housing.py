import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 

def plot_feature_importances(feature_importances, title, feature_names):

	#Normalize the importance values
	feature_importances = 100.0 * (feature_importances / max(feature_importances))

	index_sorted = np.flipud(np.argsort(feature_importances))

	#Center the location of the labels on the X-axis (for display purposes only)
	pos = np.arange(index_sorted.shape[0]) + 0.5

	#Plot the graph
	plt.figure()
	plt.bar(pos, feature_importances[index_sorted], align = 'center')
	plt.xticks(pos, feature_names[index_sorted])
	plt.ylabel("Relative importance")
	plt.title(title)
	plt.show()


housing_data = datasets.load_boston()
X, y = shuffle(housing_data.data, housing_data.target, random_state = 7)

num_training = int(len(X) * 0.8)

X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

print("Feature_names: ", housing_data.feature_names)

dt_regressor = DecisionTreeRegressor(max_depth = 4)
dt_regressor.fit(X_train, y_train)

ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 4), n_estimators = 400, random_state = 7)
ab_regressor.fit(X_train, y_train)

y_pred_dt = dt_regressor.predict(X_test)
mse = mean_squared_error(y_pred_dt, y_test)
evs = explained_variance_score(y_pred_dt, y_test)

print("\nDecision Tree Performance: ")
print("\nMean squared error = ", round(mse, 2))
print("\nExplained Variance Score = ", round(evs, 2))

y_pred_ab = ab_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_ab)
evs = explained_variance_score(y_test, y_pred_ab)

print("\nAdaBoost Performance: ")
print("\nMean squared error = ", round(mse, 2))
print("\nExplained Variance Score = ", round(evs, 2))

#Computing the relative importance of features
plot_feature_importances(dt_regressor.feature_importances_, 'DecisionTreeRegressor', housing_data.feature_names)
plot_feature_importances(ab_regressor.feature_importances_, 'AdaBoostRegressor', housing_data.feature_names)
