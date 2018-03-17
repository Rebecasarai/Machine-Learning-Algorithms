#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import os

os.chdir('C:/Users/Jeff/udacity/Intro_to_Machine_Learning/ud120-projects/naive_bayes')

import sys
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# the classifier
clf = GaussianNB()

# train
t0 = time()
clf.fit(features_train, labels_train)
print "\ntraining time:", round(time() - t0, 3), "s"

# predict
t0 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time() - t0, 3), "s"

accuracy = accuracy_score(pred, labels_test)

print '\naccuracy = {0}'.format(accuracy)