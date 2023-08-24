# -*- coding: utf-8 -*-
"""
@author: Marisa Paone
Class: CS677
Facilitator: Sarah Cameron
Date: 7/24/23
Homework#3 Problems 1-6

this scripts reads banknote data, and predicts good or bad banknotes using a simple classifier,
kNN, and logistic regression.
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection \
    import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

ticker = 'data_banknote_authentication'
here = os.path.abspath(__file__)
input_dir = os.path.abspath(os.path.join(here, os.pardir))
df = pd.read_csv(os.path.join(input_dir, ticker + '.txt'), names = ["f_1", "f_2", "f_3", "f_4", "Class"])

try:

    #-------------Question 1 Part 1-------------

    # Adding a color column green and red where green is the 0's and red is the 1's
    df["Color"] = np.where(df["Class"] == 0, 'green', 'red')
    print("---------Question 1 Part 1---------")
    data = df.describe().round(2)
    # Dropping the extra rows of df.describe
    data.drop(['count', '25%', '50%', '75%', 'max', 'min'], inplace = True)
    print(data)

    #-------------Question 1 Part 2-------------
    print()
    print("---------Question 1 Part 2---------")
    # Creating a dataframe with means and standard deviations of each feature, for greens, reds, and both
    q1_part2 = pd.DataFrame(
        {'class': [0, 1, 'all'],
         'μ(f_1)': [df.loc[df['Class'] == 0, 'f_1'].mean().round(2), df.loc[df['Class'] == 1, 'f_1'].mean().round(2), df['f_1'].mean().round(2)],
         'σ(f_1)': [df.loc[df['Class'] == 0, 'f_1'].std().round(2), df.loc[df['Class'] == 1, 'f_1'].std().round(2), df['f_1'].std().round(2)],
         'μ(f_2)': [df.loc[df['Class'] == 0, 'f_2'].mean().round(2), df.loc[df['Class'] == 1, 'f_2'].mean().round(2), df['f_2'].mean().round(2)],
         'σ(f_2)': [df.loc[df['Class'] == 0, 'f_2'].std().round(2), df.loc[df['Class'] == 1, 'f_2'].std().round(2), df['f_2'].std().round(2)],
         'μ(f_3)': [df.loc[df['Class'] == 0, 'f_3'].mean().round(2), df.loc[df['Class'] == 1, 'f_3'].mean().round(2), df['f_3'].mean().round(2)],
         'σ(f_3)': [df.loc[df['Class'] == 0, 'f_3'].std().round(2), df.loc[df['Class'] == 1, 'f_3'].std().round(2), df['f_3'].std().round(2)],
         'μ(f_4)': [df.loc[df['Class'] == 0, 'f_4'].mean().round(2), df.loc[df['Class'] == 1, 'f_4'].mean().round(2), df['f_4'].mean().round(2)],
         'σ(f_4)': [df.loc[df['Class'] == 0, 'f_4'].std().round(2), df.loc[df['Class'] == 1, 'f_4'].std().round(2), df['f_4'].std().round(2)],
         }
    )

    print(q1_part2.to_string())
    q1_part2.to_excel('Question1_Part2.xlsx')

    #-------------Question 2 Part 1-------------
    print()
    print("---------Question 2 Part 1---------")

    #splitting the dataframe into train and test sets (50% for each)
    x_train, x_test =\
        train_test_split(df, train_size = 0.5)
    print(x_train)

    #x train class of only greens/0's
    x_train_class0 = x_train[x_train['Class'] == 0]
    #creating pairplots
    green_pairplot = sns.pairplot(x_train_class0[['f_1', 'f_2', 'f_3', 'f_4']]).\
        set(title = "Good Bills")
    plt.subplots_adjust(hspace=0.2, top = 0.95)
    #saving pairplots to pdfs
    plt.savefig("good_bills.pdf", format="pdf")

    # x train class of only reds/1's
    x_train_class1 = x_train[x_train['Class'] == 1]
    red_pairplot = sns.pairplot(x_train_class1[['f_1', 'f_2', 'f_3', 'f_4']]).\
        set(title = "Fake Bills")
    plt.subplots_adjust(hspace=0.2, top = 0.95)
    plt.savefig("fake_bills.pdf", format = "pdf")

    # -------------Question 2 Part 2-------------
    # simple classifier is:
    # if f_1 > -2.5, and f_4>-2 and f_3<8 then good bill, if not then fake bill.

    # -------------Question 2 Part 3-------------
    print()
    print("---------Question 2 Part 3---------")

    prediction = x_test
    # Adding a prediction column
    prediction['Prediction'] = np.where((x_test['f_1'] > -2.5) & (x_test['f_4'] > -2) & (x_test['f_3'] < 8), 'green', 'red')
    print(prediction)

    # If the prediction is true, the value assigned will be 1, and then we can compute the actual accuracy
    # by taking the sum of the prediction row and dividing it over the total rows of the dataframe
    # Adding an accuracy column
    prediction['Accuracy'] = np.where(x_test['Prediction'] == x_test['Color'], 1, 0)
    print('Accuracy for simple classifier = ', (prediction['Accuracy'].sum()/len(prediction))*100, '%')


    # -------------Question 2 Part 4&5-------------
    print()
    print("---------Question 2 Part 4&5---------")

    def true_labels():
        # compute TP, FP, TN, FN, TPR, TNR
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        for i in x_test.index:
            if (x_test['Prediction'][i] == 'green') & ('green' == x_test['Color'][i]):
                true_positives = true_positives + 1
            elif (x_test['Prediction'][i] == 'red') & ('red' == x_test['Color'][i]):
                true_negatives = true_negatives + 1
            elif (x_test['Prediction'][i] == 'green') & ('green' != x_test['Color'][i]):
                false_positives = false_positives + 1
            else:
                false_negatives = false_negatives + 1

        #putting all calculations into a dataframe to copy and paste table from excel
        q2_part5 = pd.DataFrame(
            {
             'True Positives': [true_positives],
             'False Positives': [false_positives],
             'True Negatives': [true_negatives],
             'False Negatives': [false_negatives],
             'Accuracy': [(prediction['Accuracy'].sum()/len(prediction))],
             'TPR': [(true_positives/(true_positives+false_positives))],
             'TNR': [(true_negatives/(true_negatives+false_negatives))]
             }
        )
        #print table to console
        print(q2_part5.to_string())
        #puts table in excel
        q2_part5.to_excel('Question2_Part5.xlsx')

    true_labels()

    # -------------Question 3 Part 1,3-------------
    print()
    print("---------Question 3 Part 1&3---------")

    # Calulate k-NN with a value k
    def kNN (k):
        # for X, drop Color and Class columns from x_train, add Color column to Y
        X = x_train.drop(labels=['Color', 'Class'], axis=1)
        Y = x_train['Color']

        # for X_test, drop all extra columns (only keep features 1-4). For Y_test only include Color column.
        X_test = x_test.drop(labels=['Color', 'Class', 'Prediction', 'Accuracy'], axis=1)
        Y_test = x_test['Color']

        # Scale the data, and use a label encoder for green and red values
        scaler = StandardScaler()
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        Y_test = le.fit_transform(Y_test)
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)

        # initialize the kNN classifier and fit X and Y
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X, Y)

        #prediction of y colors using the knn classifier
        y_pred = knn_classifier.predict(X_test)

        print("Accuracy for kNN with k =", k, 'is:', accuracy_score(Y_test, y_pred))

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        for i in range(len(Y_test)):
            if (Y_test[i] == 0) & (y_pred[i] == 0):
                true_positives = true_positives + 1
            elif (Y_test[i] == 1) & (y_pred[i] == 1):
                true_negatives = true_negatives + 1
            elif (Y_test[i] == 0) & (y_pred[i] != 0):
                false_positives = false_positives + 1
            else:
                false_negatives = false_negatives + 1

        q3_part3 = pd.DataFrame(
            {
                'True Positives': [true_positives],
                'False Positives': [false_positives],
                'True Negatives': [true_negatives],
                'False Negatives': [false_negatives],
                'Accuracy': [accuracy_score(Y_test, y_pred)],
                'TPR': [(true_positives / (true_positives + false_positives))],
                'TNR': [(true_negatives / (true_negatives + false_negatives))]
            }
        )
        print(q3_part3.to_string())
        q3_part3.to_excel('Question3_Part3_k' + str(k) + '.xlsx')

        return accuracy_score(Y_test, y_pred)

    # computing the kNN method for 3, 5, 7, 9, and 11.
    k3 = kNN(3)
    k5 = kNN(5)
    k7 = kNN(7)
    k9 = kNN(9)
    k11 = kNN(11)

    # -------------Question 3 Part 2-------------
    # creating a scatter plot to visualize the best k value accuracy
    scatter_plot = plt.figure()
    axes1 = scatter_plot.add_subplot(1, 1, 1)
    axes1.scatter([3, 5, 7, 9, 11], [k3, k5, k7, k9, k11], s=100)
    axes1.set_title('Q3, P2: K Value vs. Accuracy')
    axes1.set_xlabel('K Value')
    axes1.set_ylabel('Accuracy')
    scatter_plot.show()

    # -------------Question 3 Part 5-------------
    print()
    print("---------Question 3 Part 5---------")

    #simple classifier
    # if f_1 > -2.5, and f_4>-2 and f_3<8 then good bill, if not then fake bill
    BUID = [0, 6, 4, 1] # my BUID last 4 numbers
    simple_classifier = np.where((BUID[0] > -2.5) & (BUID[3] > -2) & (BUID[2] < 8), 'green', 'red')
    print('Simple Classifier Prediction:', simple_classifier)

    #kNN
    X = x_train.drop(labels = ['Color', 'Class'], axis=1)
    Y = x_train['Color']

    X_test = x_test.drop(labels = ['Color', 'Class', 'Prediction', 'Accuracy'], axis=1)
    Y_test = x_test['Color']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    knn_classifier = KNeighborsClassifier(n_neighbors=7)
    knn_classifier.fit(X, Y)

    predict = knn_classifier.predict([BUID]) # using kNN to predict using my BUID
    print('k-NN prediction:', predict[0])

    # -------------Question 4 Part 1-------------
    print()
    print("---------Question 4 Part 1---------")

    k_values = [k3, k5, k7, k9, k11]
    # Calculates which k had the best accuracy
    def calc_max_k():
        if max(k_values) == k_values[0]:
            return 3
        elif  max(k_values) == k_values[1]:
            return 5
        elif max(k_values) == k_values[2]:
            return 7
        elif max(k_values) == k_values[3]:
            return 9
        else:
            return 11

    k_best = calc_max_k()

    # Calculates kNN that drops a feature column
    def kNN_max_drop_feature (k, feature):
        X = x_train.drop(labels=['Color', 'Class', feature], axis=1)
        Y = x_train['Color']

        X_test = x_test.drop(labels=['Color', 'Class', feature, 'Prediction', 'Accuracy'], axis=1)
        Y_test = x_test['Color']

        scaler = StandardScaler()
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        Y_test = le.fit_transform(Y_test)
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)

        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X, Y)

        y_pred = knn_classifier.predict(X_test)

        print("Accuracy for kNN with the best k =", k, 'dropping feature:', feature, 'is:', accuracy_score(Y_test, y_pred))

    kNN_max_drop_feature(k_best, 'f_1')
    kNN_max_drop_feature(k_best, 'f_2')
    kNN_max_drop_feature(k_best, 'f_3')
    kNN_max_drop_feature(k_best, 'f_4')

    # -------------Question 5 Part 1-------------
    print()
    print("---------Question 5 Part 1---------")

    def logistic_regression():
        X = x_train.drop(labels=['Color', 'Class'], axis=1)
        Y = x_train['Color']

        X_test = x_test.drop(labels=['Color', 'Class', 'Prediction', 'Accuracy'], axis=1)
        Y_test = x_test['Color']

        scaler = StandardScaler()
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        Y_test = le.fit_transform(Y_test)
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)

        # Using logistic regression to predict colors.
        log_reg_classifier = LogisticRegression()
        log_reg_classifier.fit(X, Y)

        y_pred = log_reg_classifier.predict(X_test)
        print()
        print('Accuracy for Logistic Regression is:', accuracy_score(Y_test, y_pred))

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        for i in range(len(Y_test)):
            if (Y_test[i] == 0) & (y_pred[i] == 0):
                true_positives = true_positives + 1
            elif (Y_test[i] == 1) & (y_pred[i] == 1):
                true_negatives = true_negatives + 1
            elif (Y_test[i] == 0) & (y_pred[i] != 0):
                false_positives = false_positives + 1
            else:
                false_negatives = false_negatives + 1

        q5_part1 = pd.DataFrame(
            {
                'True Positives': [true_positives],
                'False Positives': [false_positives],
                'True Negatives': [true_negatives],
                'False Negatives': [false_negatives],
                'Accuracy': [accuracy_score(Y_test, y_pred)],
                'TPR': [(true_positives / (true_positives + false_positives))],
                'TNR': [(true_negatives / (true_negatives + false_negatives))]
            }
        )
        print(q5_part1.to_string())
        q5_part1.to_excel('Question5_Part1.xlsx')

        return accuracy_score(Y_test, y_pred)

    log_reg = logistic_regression()

    # -------------Question 5 Part 5-------------
    print()
    print("---------Question 5 Part 5---------")

    #logistic regression to predict using my BUID
    X = x_train.drop(labels = ['Color', 'Class'], axis=1)
    Y = x_train['Color']

    X_test = x_test.drop(labels = ['Color', 'Class', 'Prediction', 'Accuracy'], axis=1)
    Y_test = x_test['Color']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit(X, Y)

    predict = log_reg_classifier.predict([BUID])
    print('Logistic Regression prediction:', predict[0])


    # -------------Question 6 Part 1-------------
    print()
    print("---------Question 6 Part 1---------")

    # logistic regression with dropping one feature
    def log_reg_drop_feature(feature):
        X = x_train.drop(labels=['Color', 'Class', feature], axis=1)
        Y = x_train['Color']

        X_test = x_test.drop(labels=['Color', 'Class', feature, 'Prediction', 'Accuracy'], axis=1)
        Y_test = x_test['Color']

        scaler = StandardScaler()
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        Y_test = le.fit_transform(Y_test)
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)

        log_reg_classifier = LogisticRegression()
        log_reg_classifier.fit(X, Y)

        y_pred = log_reg_classifier.predict(X_test)

        print('Accuracy for Logistic Regression dropping feature:', feature, 'is:', accuracy_score(Y_test, y_pred))

    log_reg_drop_feature('f_1')
    log_reg_drop_feature('f_2')
    log_reg_drop_feature('f_3')
    log_reg_drop_feature('f_4')

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)