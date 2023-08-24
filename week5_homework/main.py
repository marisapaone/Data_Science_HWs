# -*- coding: utf-8 -*-
"""
@author: Marisa Paone
Class: CS677
Facilitator: Sarah Cameron
Date: 8/7/23
Homework#5 Problems 1-5

this script compares Naive Bayesian, Decision Tree and Random Forest classifications for identifying
normal/non-normal fetus status based on fetal cardiograms.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection \
    import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn import tree

from sklearn.naive_bayes import GaussianNB

file_name = 'CTG.xls'
sheet = 'Raw Data'
here = os.path.abspath(__file__)
input_dir = os.path.abspath(os.path.join(here, os.pardir))
df = pd.read_excel(os.path.join(input_dir, file_name), sheet_name= sheet)

try:
    # -------------Question 1-------------
    print('---------Question 1---------')
    print(df)

    df_subset = df.copy()

    df_subset = df[['NSP', 'LB', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Min', 'Max', 'Mode', 'Mean', 'Median', 'Variance']]
    # drops 'not a number' rows
    df_subset = df_subset.dropna()

    # fills in 0 for 2's and 3's which indicate abnormal.
    # 1's are normal.
    df_subset['Class'] = np.where(df_subset['NSP'] == 1, 1, 0)
    print(df_subset)

    # ------- Question 2 ----------
    print()
    print('---------Question 2---------')

    X = df_subset[['ASTV', 'MLTV', 'Max', 'Median']].values
    Y = df_subset['Class'].values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.5)

    NB_classifier = GaussianNB().fit(x_train, y_train)
    prediction_NB = NB_classifier.predict(x_test)

    print('Accuracy for Naive Bayes: ', accuracy_score(y_test, prediction_NB))

    def calc_rates(cm):
        TP = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TN = cm[1, 1]
        TPR = (TP / (TP + FN))
        TNR = (TN / (FP + TN))
        print('TPR = ', TPR)
        print('TNR = ', TNR)

    # Calculating confusion matrix
    cm_NB = confusion_matrix(y_test, prediction_NB)
    print(cm_NB)
    calc_rates(cm_NB)

    # ------------Question 3-------------
    print()
    print('---------Question 3---------')

    DT_classifier = tree.DecisionTreeClassifier()
    DT_classifier = DT_classifier.fit(x_train, y_train)
    prediction_DT = DT_classifier.predict(x_test)
    print('Accuracy for Decision Tree: ', accuracy_score(y_test, prediction_DT))

    # Calculating confusion matrix
    cm_DT = confusion_matrix(y_test, prediction_DT)
    print(cm_DT)
    calc_rates(cm_DT)

    #------------Question 4-------------
    print()
    print('---------Question 4---------')

    data = pd.DataFrame(columns = ['estimators', 'depth', 'error rate', 'accuracy', 'pred'])
    print('Random Forest: ')

    for i in range (10):
        for j in range (5):
            RF_classifier = RandomForestClassifier(n_estimators=i+1, max_depth=j+1, criterion='entropy')
            RF_classifier.fit(x_train, y_train)
            prediction_RF = RF_classifier.predict(x_test)
            error_rate = np.mean(prediction_RF != y_test)
            #print('Accuracy for Random Forest with', i+1, 'estimators and a max depth of', j+1, ':', accuracy_score(y_test, prediction_RF))
            #print('Error rate with', i+1, 'estimators and a max depth of', j+1, ':', error_rate)
            new_row = {'estimators':i+1,'depth': j+1,  'error rate':error_rate, 'accuracy': accuracy_score(y_test, prediction_RF), 'pred': prediction_RF}
            data.loc[len(data)] = new_row

    #printing entire random forest dataframe (without predictions)
    print(data.iloc[:,[0,1,2,3]])

    # plotting error rates
    one = data[data['depth'] == 1]
    two = data[data['depth'] == 2]
    three = data[data['depth'] == 3]
    four = data[data['depth'] == 4]
    five = data[data['depth'] == 5]

    plt.plot(one['estimators'], one['error rate'], color = 'red', marker = 'o', label = 'max depth = 1')
    plt.plot(two['estimators'], two['error rate'], color='green', marker='o', label = 'max depth = 2')
    plt.plot(three['estimators'], three['error rate'], color='blue', marker='o',label = 'max depth = 3')
    plt.plot(four['estimators'], four['error rate'], color='purple', marker='o', label = 'max depth = 4')
    plt.plot(five['estimators'], five['error rate'], color='orange', marker='o', label = 'max depth = 5')
    plt.title('Random Forest for ASTV and MLTV', fontsize = 10)
    plt.xlabel('n_estimators')
    plt.ylabel('error rate')
    plt.legend(fontsize = 8)
    plt.grid(True)
    plt.show()

    # Calculating best accuracy row and lowest error row (they are always the same)
    best_accuracy = data.iloc[[data['accuracy'].idxmax()]]
    print('Best Accuracy Row: \n', best_accuracy.iloc[:, [0,1,2,3]])
    lowest_error = data.iloc[[data['error rate'].idxmin()]]
    print('Lowest Error Row: \n',lowest_error.iloc[:, [0,1,2,3]])

    # Calculating confusion matrix
    cm_RF = confusion_matrix(y_test, best_accuracy['pred'].iloc[0])
    print('Confusion matrix for n =',int(best_accuracy['estimators'].iloc[0]), 'and depth =', int(best_accuracy['depth'].iloc[0]), '\n', cm_RF)

    calc_rates(cm_RF)


except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', file_name)