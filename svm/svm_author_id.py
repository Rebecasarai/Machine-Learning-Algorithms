#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
from sklearn.metrics import accuracy_score

from time import time
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


def my_svm(features_train, features_test, labels_train, labels_test, kernel='linear', C=1.0):
    # the classifier
    clf = SVC(kernel=kernel, C=C)

    # train
    t0 = time()
    clf.fit(features_train, labels_train)
    print "\ntraining time:", round(time()-t0, 3), "s"

    # predict
    t0 = time()
    pred = clf.predict(features_test)
    print "predicting time:", round(time()-t0, 3), "s"

    accuracy = accuracy_score(pred, labels_test)

    print '\naccuracy = {0}'.format(accuracy)
    return pred

pred = my_svm(features_train, features_test, labels_train, labels_test)