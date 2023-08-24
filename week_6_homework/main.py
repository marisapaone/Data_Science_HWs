# -*- coding: utf-8 -*-
"""
@author: Marisa Paone
Class: CS677
Facilitator: Sarah Cameron
Date: 8/10/23
Homework #6

This script looks into seed data and k means clustering and compares it to SVM and Naive Bayesian

"""
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection \
    import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ticker = 'seeds_dataset.csv'
here = os.path.abspath(__file__)
input_dir = os.path.abspath(os.path.join(here, os.pardir))
df = pd.read_csv(os.path.join(input_dir, ticker), sep='\t', names = ['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'Asymmetry Coefficent', 'Groove Length', 'Class'])

try:

    print(df)
    # only include class 1 and 3
    df_subset = df[df['Class'] != 2]

    print(df_subset)
    # scaling the data
    X = df_subset[['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'Asymmetry Coefficent', 'Groove Length']].values
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    Y = df_subset['Class'].values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.5)


    def calc_rates(cm):
        TP = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TN = cm[1, 1]
        TPR = (TP / (TP + FN))
        TNR = (TN / (FP + TN))
        print('TPR = ', TPR)
        print('TNR = ', TNR)


    print('\n-----------Question 1------------')

    # Linear SVM
    print('\n Linear SVM')

    svm_classifier = svm.SVC(kernel = 'linear')
    svm_classifier.fit(x_train, y_train)

    predict_lin = svm_classifier.predict(x_test)#.reshape(-1, 1)

    score = svm_classifier.score(x_test, y_test)
    accuracy = metrics.accuracy_score(y_test, predict_lin)
    print(score)
    print(accuracy)

    cm_lin = confusion_matrix(y_test, predict_lin)
    print(cm_lin)
    calc_rates(cm_lin)

    # Guassian SVM
    print('\n Gaussian SVM')

    gaus_svm_classifier = svm.SVC(kernel='rbf')
    gaus_svm_classifier.fit(X,Y)

    predict_gaus = gaus_svm_classifier.predict(x_test)
    score = gaus_svm_classifier.score(x_test, y_test)
    accuracy = metrics.accuracy_score(y_test, predict_gaus)
    print(score)
    print(accuracy)
    cm_gaus = confusion_matrix(y_test, predict_gaus)
    print(cm_gaus)
    calc_rates(cm_gaus)

    #polynomial svm
    print('\n Polynomial SVM')

    poly_svm_classifier = svm.SVC(kernel='poly', degree=3)
    poly_svm_classifier.fit(X, Y)

    predict_poly = poly_svm_classifier.predict(x_test)
    score = poly_svm_classifier.score(x_test, y_test)
    accuracy = metrics.accuracy_score(y_test, predict_poly)
    print(score)
    print(accuracy)
    cm_poly = confusion_matrix(y_test, predict_poly)
    print(cm_poly)
    calc_rates(cm_poly)

    print('\n-----------Question 2------------')

    # My Classifier: Naive Bayesian
    print('\n Naive Bayesian')
    NB_classifier = GaussianNB().fit(x_train, y_train)
    prediction_NB = NB_classifier.predict(x_test)

    print('Accuracy for Naive Bayes: ', accuracy_score(y_test, prediction_NB))
    cm_NB = confusion_matrix(y_test, prediction_NB)
    print(cm_NB)
    calc_rates(cm_NB)

    # Question 3 K means clustering
    print('\n-----------Question 3------------')

    # using lowercase x and y for the entire dataset
    x = df[['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'Asymmetry Coefficent', 'Groove Length']].values
    y = df['Class'].values
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    # to remove the init random warnings
    import warnings
    warnings.filterwarnings("ignore")

    # looking at k's from 1 to 8
    K = range(1,9)
    inertia_list = []
    for k in K:
        k_means = KMeans(n_clusters=k, init = 'random')
        y_kmeans = k_means.fit_predict(x)
        inertia = k_means.inertia_
        inertia_list.append(inertia)

    # plotting distortion
    fig, ax = plt.subplots(1,figsize = (5,5))
    plt.plot(K, inertia_list, marker = 'o', color = 'blue')
    plt.title('Knee Method: k vs inertia')
    plt.xlabel('number of clusters k')
    plt.ylabel('inertia')
    plt.tight_layout()
    plt.savefig('Knee Method')
    plt.show()

    # the best number of clusters was 3
    k_means = KMeans(n_clusters=3)
    y_kmeans = k_means.fit_predict(x)
    centroids = k_means.cluster_centers_
    print('Centroids:', centroids)

    # generating two random ints from 0 to 6 (7 feature values)
    int1 =random.randint(0,6)
    int2 = random.randint(0,6)
    while int2 == int1:
        int2 = random.randint(0,6)

    # printing out the ints for readability
    print()
    print("ints", int1, int2)

    # getting a list of feature names (column names)
    feature_names = df.columns.values.tolist()

    # plotting the clustering
    fig, ax = plt.subplots(1, figsize=(7,5))
    plt.scatter(x[y_kmeans == 0, int1], x[y_kmeans == 0, int2], s=75, c = 'red', label = 'Kama 1')
    plt.scatter(x[y_kmeans == 1, int1], x[y_kmeans == 1, int2], s=75, c='blue', label='Rosa 2')
    plt.scatter(x[y_kmeans == 2, int1], x[y_kmeans == 2, int2], s=75, c='green', label='Canadian 3')
    plt.scatter(centroids[:, int1], centroids[:,int2], s=200, c='black', label='Centroids')
    plt.legend()
    plt.title('K-means Clustering for Seeds')
    plt.xlabel(feature_names[int1])
    plt.ylabel(feature_names[int2])
    plt.tight_layout()
    plt.savefig('K-means Clustering for Seeds')
    plt.show()

    # taking the clusters x and y values from the two feature graph
    centroids_mean = [centroids[0,int1], centroids [0,int2]], [centroids[1,int1], centroids[1,int2]], [centroids[2,int1], centroids[2,int2]]
    print('Kama Centroid:', centroids_mean[0])
    print('Rosa Centroid:', centroids_mean[1])
    print('Canadian Centroid:', centroids_mean[2])
    print(centroids_mean)
    # labeling the clusters since their means were written in order
    centroids_labels = [1, 2, 3]

    # part 4
    # using the two features for kNN with k=1
    print('\n Question 3, Part 4: Using the two Random Features:')
    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    x = df[[feature_names[int1], feature_names[int2]]].values
    y = df['Class'].values
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    knn_classifier.fit(centroids_mean, centroids_labels)
    y_pred = knn_classifier.predict(x)
    print(y_pred)
    print(y)
    print('Accuracy:',accuracy_score(y,y_pred))

    # part 4 using all 7 features
    # using all features for kNN with k=1
    print('\n Question 3, Part 4: Using all 7 Features:')
    x = df[['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'Asymmetry Coefficent', 'Groove Length']].values
    y = df['Class'].values
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    knn_classifier.fit(centroids, centroids_labels)
    y_pred = knn_classifier.predict(x)
    print(y_pred)
    print(y)
    print('Accuracy:', accuracy_score(y, y_pred))


    # part 5 with only two features
    print('\n Question 3, Part 5: Using the two random features:')
    centroids_mean = [centroids[0, int1], centroids[0, int2]],  [centroids[2, int1], centroids[2, int2]]
    # Removed label 2
    centroids_labels = [1, 3]
    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    x_subset = df_subset[[feature_names[int1], feature_names[int2]]].values
    y_subset = df_subset['Class'].values
    scaler = StandardScaler()
    scaler.fit(x_subset)
    x_subset = scaler.transform(x_subset)

    knn_classifier.fit(centroids_mean, centroids_labels)
    y_pred_subset = knn_classifier.predict(x_subset)
    print(y_pred_subset)
    print(Y)
    print('Accuracy:', accuracy_score(y_subset, y_pred_subset))
    cm_2feat = confusion_matrix(y_subset, y_pred_subset)
    print(cm_2feat)

    # part 5 using all 7 features
    print('\n Question 3, Part 5: Using all 7 Features:')
    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    centroids = [centroids[0, :], centroids[2, :]]
    knn_classifier.fit(centroids, centroids_labels)
    y_pred_subset = knn_classifier.predict(X)
    print(y_pred_subset)
    print(Y)
    print('Accuracy:', accuracy_score(Y, y_pred_subset))
    cm_7feat = confusion_matrix(Y, y_pred_subset)
    print(cm_7feat)


except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)