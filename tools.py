# https://github.com/amueller/introduction_to_ml_with_python/tree/master/mglearn

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


cm = ListedColormap(['#0000aa', '#ff2020'])


def visualize_coefficients(coefficients, feature_names, n_top_features=25):
	"""Visualize coefficients of a linear model.
	Parameters
	----------
	coefficients : nd-array, shape (n_features,)
			Model coefficients.
	feature_names : list or nd-array of strings, shape (n_features,)
			Feature names for labeling the coefficients.
	n_top_features : int, default=25
			How many features to show. The function will show the largest (most
			positive) and smallest (most negative)  n_top_features coefficients,
			for a total of 2 * n_top_features coefficients.
	"""
	coefficients = coefficients.squeeze()
	if coefficients.ndim > 1:
		# this is not a row or column vector
		raise ValueError("coeffients must be 1d array or column vector, got shape {}".format(coefficients.shape))

	coefficients = coefficients.ravel()

	if len(coefficients) != len(feature_names):
		raise ValueError("Number of coefficients {} doesn't match number of feature names {}.".format(
			len(coefficients),
			len(feature_names)
		))
	# get coefficients with large absolute values

	coef = coefficients.ravel()
	positive_coefficients = np.argsort(coef)[-n_top_features:]
	negative_coefficients = np.argsort(coef)[:n_top_features]
	interesting_coefficients = np.hstack([negative_coefficients, positive_coefficients])

	# plot them
	plt.figure(figsize=(15, 7))
	colors = [cm(1) if c < 0 else cm(0) for c in coef[interesting_coefficients]]
	plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients], color=colors)
	feature_names = np.array(feature_names)
	plt.subplots_adjust(bottom=0.3)
	plt.xticks(np.arange(1, 1 + 2 * n_top_features), feature_names[interesting_coefficients], rotation=90, ha="right", fontsize=9)
	plt.ylabel("Coefficient magnitude")
	plt.xlabel("Feature")
