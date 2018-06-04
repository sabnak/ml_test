from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

def indices_of_top_k(arr, k):
	return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
	def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
		self.add_bedrooms_per_room = add_bedrooms_per_room

	def fit(self, X, y=None):
		return self  # nothing else to do

	def transform(self, X, y=None):
		rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
		population_per_household = X[:, population_ix] / X[:, household_ix]
		if self.add_bedrooms_per_room:
			bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
			return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
		else:
			return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return X[self.attribute_names].values


class TopFeatureSelector(BaseEstimator, TransformerMixin):

	def __init__(self, feature_list, k):
		self.feature_list = feature_list
		self.k = k

	def fit(self, X, y=None):
		self.feature_indices_ = indices_of_top_k(self.feature_list, self.k)
		return self

	def transform(self, X, y=None):
		return X[:, self.feature_indices_]