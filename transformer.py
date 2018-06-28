from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import re
from html import unescape
import nltk
import urlextract

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

stemmer = nltk.PorterStemmer()
url_extractor = urlextract.URLExtract()


def indices_of_top_k(arr, k):
	return np.sort(np.argpartition(np.array(arr), -k)[-k:])


def html_to_plain_text(html):

	replacement_list = [
		(r"<head.*?>.*?</head>", ""),
		(r"<a\s.*?>", " HYPERLINK "),
		(r"<.*?>", ""),
		(r"(\s*\n)+", "\n"),
	]

	for (pattern, replacement) in replacement_list:
		html = re.sub(pattern, replacement, html, flags=re.S | re.I)

	return unescape(html)


def email_to_text(email, unify_urls=True, unify_numbers=True):

	try:
		from_ = email.get("from")
	except IndexError:
		from_ = None

	domain = re.search("@(.+)", from_) if from_ else None

	try:
		subject = email.get("subject")
	except IndexError:
		subject = None

	content = "{} {} {} ".format(
		from_ if from_ else "EMPTYFROM",
		domain if domain else "EMPTYDOMAIN",
		subject if subject else "EMPTYSUBJECT"
	)

	is_html = False

	for part in email.walk():
		content_type = part.get_content_type()

		if content_type not in ("text/plain", "text/html"):
			continue

		try:
			content += part.get_content()
		except:
			content += str(part.get_payload())

		if content_type == "text/html":
			is_html = True

		break

	content = content.strip()

	if not content:
		return ""

	if is_html:
		content = html_to_plain_text(content)

	if unify_numbers:
		content = re.sub(r"[\d][\d.]*", " NUMBER ", content)

	if unify_urls:
		urls = list(set(url_extractor.find_urls(content)))
		urls.sort(key=lambda url: len(url), reverse=True)
		for url in urls:
			content = content.replace(url, " URL ")

	return str(content)


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

