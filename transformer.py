from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy.sparse import csr_matrix
import re
from html import unescape
from collections import Counter
import nltk
import urlextract

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

stemmer = nltk.PorterStemmer()
url_extractor = urlextract.URLExtract()


def indices_of_top_k(arr, k):
	return np.sort(np.argpartition(np.array(arr), -k)[-k:])


def html_to_plain_text(html):
	text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
	text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
	text = re.sub('<.*?>', '', text, flags=re.M | re.S)
	text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
	return unescape(text)


def email_to_text(email):
	html = None
	for part in email.walk():
		ctype = part.get_content_type()
		if not ctype in ("text/plain", "text/html"):
			continue
		try:
			content = part.get_content()
		except:  # in case of encoding issues
			content = str(part.get_payload())
		if ctype == "text/plain":
			return content
		else:
			html = content
	if html:
		return html_to_plain_text(html)


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


class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):

	def __init__(
			self,
			strip_headers=True,
			lower_case=True,
			remove_punctuation=True,
			replace_urls=True,
			replace_numbers=True,
			stemming=True
	):
		self.strip_headers = strip_headers
		self.lower_case = lower_case
		self.remove_punctuation = remove_punctuation
		self.replace_urls = replace_urls
		self.replace_numbers = replace_numbers
		self.stemming = stemming

	def fit(self, X, y=None):
		return self

	def transform(self, X, y=None):
		X_transformed = []

		for data in X:
			text = email_to_text(data) or ""

			if self.lower_case:
				text = text.lower()

			if self.replace_urls and url_extractor is not None:
				urls = list(set(url_extractor.find_urls(text)))
				urls.sort(key=lambda url: len(url), reverse=True)
				for url in urls:
					text = text.replace(url, " URL ")

			if self.replace_numbers:
				text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)

			if self.remove_punctuation:
				text = re.sub(r'\W+', ' ', text, flags=re.M)

			word_counts = Counter(text.split())

			if self.stemming and stemmer is not None:
				stemmed_word_counts = Counter()

				for word, count in word_counts.items():
					stemmed_word = stemmer.stem(word)
					stemmed_word_counts[stemmed_word] += count

				word_counts = stemmed_word_counts

			X_transformed.append(word_counts)

		return np.array(X_transformed)


class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, vocabulary_size=1000):
		self.vocabulary_size = vocabulary_size

	def fit(self, X, y=None):
		total_count = Counter()
		for word_count in X:
			for word, count in word_count.items():
				total_count[word] += min(count, 10)
		most_common = total_count.most_common()[:self.vocabulary_size]
		self.most_common_ = most_common
		self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
		return self

	def transform(self, X, y=None):
		rows = []
		cols = []
		data = []
		for row, word_count in enumerate(X):
			for word, count in word_count.items():
				rows.append(row)
				cols.append(self.vocabulary_.get(word, 0))
				data.append(count)
		return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))
