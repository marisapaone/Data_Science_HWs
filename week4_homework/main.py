# -*- coding: utf-8 -*-
"""
@author: Marisa Paone
Class: CS677
Facilitator: Sarah Cameron
Date: 7/27/23
Homework#4 Problems 1-3

this script looks at health data of patients who had heart failure, it looks into 4 features and graphs linear
quadratic, cubic, and logarithmic regression of the data.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection \
    import train_test_split
import seaborn as sns

ticker = 'heart_failure_clinical_records_dataset'
here = os.path.abspath(__file__)
input_dir = os.path.abspath(os.path.join(here, os.pardir))
df = pd.read_csv(os.path.join(input_dir, ticker + '.csv'))

try:
    # -------------Question 1-------------
    print('-------------Question 1-------------')
    print(df)

    # selecting columns we would like to include
    df_select = df[['creatinine_phosphokinase', 'serum_creatinine', 'serum_sodium', 'platelets', 'DEATH_EVENT']]
    df_1 = df_select.loc[df_select['DEATH_EVENT'] == 1]
    df_0 = df_select.loc[df_select['DEATH_EVENT'] == 0]

    print(df_0)
    print(df_1)

    df_0_rem = df_0.drop(columns = ['DEATH_EVENT'])
    df_1_rem = df_1.drop(columns = ['DEATH_EVENT'])

    # correlation graphs

    corr_df = df_0_rem.corr(method = 'pearson')
    plt.figure(figsize=(12,10))
    plt.title('Surviving Patients', y=1, pad=20, size=20)
    sns.heatmap(corr_df, annot = True)
    plt.savefig('Death_Event_0.png', bbox_inches='tight')
    plt.show()

    corr_df_1 = df_1_rem.corr(method = 'pearson')
    plt.figure(figsize=(12,10))
    plt.title('Deceased Patients', y=1, pad=20, size=20)
    sns.heatmap(corr_df_1, annot = True)
    plt.savefig('Death_Event_1.png', bbox_inches='tight')
    plt.show()

    # -------------Question 2-------------

    print('-------------Question 2-------------')

    # Group 2
    # X = platelets
    # Y = Serum Sodium

    # SURVIVING PATIENTS X and Y dataframes
    X = df_0['platelets']
    Y = df_0['serum_sodium']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.5)
    # DESEASED PATIENTS X and Y dataframes
    X_1 = df_1['platelets']
    Y_1 = df_1['serum_sodium']
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(X_1, Y_1, train_size=0.5)

    # creates scatter plots with a regression line and prints the regression lines equation for DEATH EVENT = 0.
    # you can change the degree to indicate quadratic, cubic, etc.
    def regression_surviving(title, degree):
        print(title)
        weights = np.polyfit(x_train, y_train, degree)
        print("Weights:", weights)
        model = np.poly1d(weights)
        y_predicted = model(x_test)
        residual = (y_test - y_predicted)

        sse = sum((residual)**2)
        print("Sum of the squared residuals for degree", degree, ' = ', sse)

        ax, fig = plt.subplots()
        plt.scatter(x_test, y_predicted, color='red', label = 'Prediction')
        plt.scatter(x_test, y_test, color = 'green', label = 'Test Set')

        x_new = np.linspace(x_test.min(), x_test.max(), 500)

        if degree == 1:
            equation = ' ' + weights[0].astype('str') + 'x + ' + weights[1].astype('str')
            print('Equation: ' + equation)
            plt.plot(x_new, ((weights[0] * x_new) + weights[1]))
        elif degree == 2:
            equation = weights[0].astype('str') + 'x^2 + ' + weights[1].astype('str') + 'x + ' + weights[2].astype('str')
            print('Equation: ' + equation)
            plt.plot(x_new, ((weights[0] * x_new**2) + (weights[1]*x_new) + weights[2]))
        elif degree == 3:
            equation = weights[0].astype('str') + 'x^3 + ' + weights[1].astype('str') + 'x^2 + ' + weights[2].astype('str') + 'x + ' + weights[3].astype('str')
            print('Equation: ' + equation)
            plt.plot(x_new, ((weights[0] * x_new ** 3) + (weights[1] * x_new**2) + (weights[2]*x_new) + weights[3]))

        plt.title(title)
        plt.xlabel("platelets")
        plt.ylabel("serum sodium")
        plt.legend()
        plt.savefig(title+'.png')
        plt.show()


    # creates scatter plots with a regression line and prints the regression lines equation for DEATH EVENT =1.
    # you can change the degree to indicate quadratic, cubic, etc.
    def regression_deceased(title, degree):
        print(title)
        weights = np.polyfit(x_train_1, y_train_1, degree)
        print("Weights:", weights)
        model = np.poly1d(weights)
        y_predicted_1 = model(x_test_1)
        residual = (y_test_1 - y_predicted_1)

        sse = sum((residual)**2)
        print("Sum of the squared residuals for degree", degree, ' = ', sse)

        plt.scatter(x_test_1, y_predicted_1, color='red', label = 'Prediction')
        plt.scatter(x_test_1, y_test_1, color = 'green', label = 'Test Set')
        x_new = np.linspace(x_test_1.min(), x_test_1.max(), 500)

        if degree == 1:
            equation = ' ' + weights[0].astype('str') + 'x + ' + weights[1].astype('str')
            print('Equation: ' + equation)
            plt.plot(x_new, (weights[0] * x_new) + weights[1], label = weights[0])
        elif degree == 2:
            equation = weights[0].astype('str') + 'x^2 + ' + weights[1].astype('str') + 'x + ' + weights[2].astype(
                'str')
            print('Equation: ' + equation)
            plt.plot(x_new, (weights[0] * x_new**2) + (weights[1]*x_new) + weights[2])
        elif degree == 3:
            equation = weights[0].astype('str') + 'x^3 + ' + weights[1].astype('str') + 'x^2 + ' + weights[2].astype(
                'str') + 'x + ' + weights[3].astype('str')
            print('Equation: ' + equation)
            plt.plot(x_new, (weights[0] * x_new ** 3) + (weights[1] * x_new**2) + (weights[2]*x_new) + weights[3])

        plt.title(title)
        plt.xlabel("platelets")
        plt.ylabel("serum sodium")
        plt.legend()
        plt.savefig(title + '.png')
        plt.show()

    regression_surviving("Simple Linear Regression Surviving Patients", 1)
    print()
    regression_deceased("Simple Linear Regression Deceased Patients", 1)
    print()
    regression_surviving("Quadratic Regression Surviving Patients", 2)
    print()
    regression_deceased("Quadratic Regression Deceased Patients", 2)
    print()
    regression_surviving("Cubic Spline Regression Surviving Patients", 3)
    print()
    regression_deceased("Cubic Spline Regression Deceased Patients", 3)
    print()


    # Calculates and plots the logarithmic function y = alogx + b for DEATH EVENT = 0
    def logarithmic_x_surviving(title):
        print(title)
        weights = np.polyfit(np.log(x_train), y_train, 1)
        print("Weights:", weights)
        model = np.poly1d(weights)
        y_predicted = model(np.log(x_test))
        residual = (y_test - y_predicted)

        sse = sum((residual) ** 2)
        print("Sum of the squared residuals = ", sse)
        equation = 'y = ' + weights[0].astype('str') + 'log(x) + ' + weights[1].astype('str')
        print('Equation: ' + equation)

        plt.scatter(x_test, y_predicted, color='orange', label = 'Prediction')
        plt.scatter(x_test, y_test, color='darkslategrey', label = 'Test Set')
        # makes a line through the points
        x_new = np.linspace(x_test.min(), x_test.max(), 500)
        plt.plot(x_new, (weights[0] * np.log(x_new)) + weights[1])

        plt.title(title)
        plt.xlabel("platelets")
        plt.ylabel("serum sodium")
        plt.legend()
        plt.savefig(title + '.png')
        plt.show()


    # Calculates and plots the logarithmic function logy = alogx + b for DEATH EVENT = 0
    def logarithmic_xy_surviving(title):
        print(title)
        weights = np.polyfit(np.log(x_train), np.log(y_train), 1)
        print("Weights:", weights)
        model = np.poly1d(weights)
        y_predicted = model(np.log(x_test))
        residual = (y_test - np.exp(y_predicted))

        sse = sum((residual) ** 2)
        print("Sum of the squared residuals = ", sse)
        equation = 'log(y) = ' + weights[0].astype('str') + 'log(x) + ' + weights[1].astype('str')
        print('Equation: ' + equation)

        plt.scatter(x_test, np.exp(y_predicted), color='orange', label = 'Prediction')
        plt.scatter(x_test, y_test, color='darkslategrey', label = 'Test Set')

        x_new = np.linspace(x_test.min(), x_test.max(), 500)
        plt.plot(x_new, np.exp((weights[0] * np.log(x_new)) + weights[1]))

        plt.title(title)
        plt.xlabel("platelets")
        plt.ylabel("serum sodium")
        plt.legend()
        plt.savefig(title + '.png')
        plt.show()

    # Calculates and plots the logarithmic function y = alogx + b for DEATH EVENT = 1
    def logarithmic_x_deceased(title):
        print(title)
        weights = np.polyfit(np.log(x_train_1), y_train_1, 1)
        print("Weights:", weights)
        model = np.poly1d(weights)
        y_predicted_1 = model(np.log(x_test_1))
        residual = (y_test_1 - y_predicted_1)

        sse = sum((residual)**2)
        print("Sum of the squared residuals = ", sse)

        equation = 'y = ' + weights[0].astype('str') + 'log(x) + ' + weights[1].astype('str')
        print('Equation: ' + equation)

        plt.scatter(x_test_1, y_predicted_1, color='orange', label = 'Prediction')
        plt.scatter(x_test_1, y_test_1, color = 'darkslategrey', label = 'Test Set')

        x_new = np.linspace(x_test_1.min(), x_test_1.max(), 500)
        plt.plot(x_new, (weights[0] * np.log(x_new)) + weights[1])

        plt.title(title)
        plt.xlabel("platelets")
        plt.ylabel("serum sodium")
        plt.legend()
        plt.savefig(title + '.png')
        plt.show()


    # Calculates and plots the logarithmic function logy = alogx + b for DEATH EVENT = 1
    def logarithmic_xy_deceased(title):
        print(title)
        weights = np.polyfit(np.log(x_train_1), np.log(y_train_1), 1)
        print("Weights:", weights)
        model = np.poly1d(weights)
        y_predicted_1 = model(np.log(x_test_1))

        residual = (y_test_1 - np.exp(y_predicted_1))
        sse = sum((residual)**2)
        print("Sum of the squared residuals = ", sse)

        equation = 'log(y) = ' + weights[0].astype('str') + 'log(x) + ' + weights[1].astype('str')
        print('Equation: ' + equation)

        plt.scatter(x_test_1, np.exp(y_predicted_1), color='orange', label = 'Prediction')
        plt.scatter(x_test_1, y_test_1, color = 'darkslategrey', label = 'Test Set')

        x_new = np.linspace(x_test_1.min(), x_test_1.max(), 500)
        plt.plot(x_new, np.exp((weights[0]*np.log(x_new))+ weights[1]))

        plt.xlabel("platelets")
        plt.ylabel("serum sodium")
        plt.legend()
        plt.title(title)
        plt.savefig(title + '.png')
        plt.show()


    logarithmic_x_surviving("GLM X Logarithmic Regression of Survivors")
    print()
    logarithmic_xy_surviving("GLM XY Logarithmic Regression of Survivors")
    print()
    logarithmic_x_deceased("GLM X Logarithmic Regression of the deceased")
    print()
    logarithmic_xy_deceased("GLM XY Logarithmic Regression of the deceased")

    # -------------Question 3-------------
    # See pdf!

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)