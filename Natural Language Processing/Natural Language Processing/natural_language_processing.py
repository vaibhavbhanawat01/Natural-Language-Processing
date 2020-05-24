# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:38:20 2020

@author: vaibhav_bhanawat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as mat

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '	', quoting = 3)


import re
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
corpos = []
for i in range (0, 1000):
    review  = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review  = review.lower()
    review = review.split()
    porterStem = PorterStemmer()
    review  = [porterStem.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpos.append(review)


#create bag of words model
from sklearn.feature_extraction.text import CountVectorizer
countVector = CountVectorizer(max_features = 1500)
X = countVector.fit_transform(corpos).toarray()
Y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

from sklearn.svm import SVC 
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)