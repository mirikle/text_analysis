# Reference: http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# Naive bayes.
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

categories = ['alt.atheism', 'soc.religion.christian',
  'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',
  categories = categories, shuffle = True, random_state=42)

phaseByPhaseTrain = False
# Phase by phase detailed illustration.
if phaseByPhaseTrain:
  print("\n".join(twenty_train.data[0].split("\n")[:3]))
  print(twenty_train.target_names[twenty_train.target[0]])

  # Bag of words.
  countVectorizer = CountVectorizer()
  xTrainCounts = countVectorizer.fit_transform(twenty_train.data)
  print xTrainCounts.shape

  print countVectorizer.vocabulary_.get(u'algorithm')

  # Transform the bag of words to also consider the length of the article.
  tfTransformer = TfidfTransformer(use_idf = False).fit(xTrainCounts)
  xTrainTfidf = tfTransformer.transform(xTrainCounts)

  print xTrainTfidf.shape

  textClassifier = MultinomialNB().fit(xTrainTfidf, twenty_train.target)
  newDocs = ['God is love', 'OpenGL on the GPU is fast.']
  newDocsCounts = countVectorizer.transform(newDocs)
  xNewTfidf = tfTransformer.transform(newDocsCounts)

  predicted = textClassifier.predict(xNewTfidf)

  for doc, category in zip(newDocs, predicted):
    print ('%r => %s' % (doc, twenty_train.target_names[category]))

# Use pipeline.
else:
  # You can also use a pipeline
  from sklearn.pipeline import Pipeline
  textClassifier = Pipeline([ ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss = 'hinge', penalty = 'l2')) ])
    # ('clf', MultinomialNB())])
  textClassifier.fit(twenty_train.data, twenty_train.target)

  import numpy as np
  twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)
  predicted = textClassifier.predict(twenty_test.data)
  print predicted == twenty_test.target
  print 'Error:', np.mean(predicted == twenty_test.target)

  # Metrics.
  from sklearn import metrics
  print(metrics.classification_report(twenty_test.target, predicted,
    target_names = twenty_test.target_names))
  print(metrics.confusion_matrix(twenty_test.target, predicted))

  # Grid search for hyper parameters.
  from sklearn.grid_search import GridSearchCV
  # Prefix name here is the pipeline phase name in line 48.
  parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3)
  }
  gridSearchClassifier = GridSearchCV(textClassifier, parameters, n_jobs = -1)
  gridSearchClassifier.fit(twenty_train.data[:600], twenty_train.target[:600])

  bestParameters, score, _ = max(gridSearchClassifier.grid_scores_, key = lambda x: x[1])
  for paramName in sorted(parameters.keys()):
    print("%s: %r" % (paramName, bestParameters[paramName]))



