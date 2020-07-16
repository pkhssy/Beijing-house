#train
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def drop(D,str): return D.drop(columns=[str])
def concat(D1,D2): return pd.concat([D1,D2], axis=1)
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

data = pd.read_csv("./csv/preprocessed.csv")

train_set, test_set = split_train_test(data, 0.2)
print (len(train_set), "train +", len(test_set), "test")
train_set_x, test_set_x = drop(train_set,'totalPrice'), drop(test_set,'totalPrice')
train_set_y, test_set_y = train_set['totalPrice'], test_set['totalPrice']

def Regression_(reg_):
    reg_.fit(train_set_x, train_set_y)
    MSE(reg_,train_set_x,train_set_y,test_set_x,test_set_y)

#Mean Squared Error
def MSE(fit,train_set_x,train_set_y,test_set_x,test_set_y):
    predict_y = fit.predict(test_set_x)
    print("MSE: %.2f" % np.sqrt(np.mean((predict_y - test_set_y) ** 2)))
    # Explained variance score: 1 is perfect prediction
    print('Train Variance score: %.2f' % fit.score(train_set_x, train_set_y))
    print('Test Variance score: %.2f' % fit.score(test_set_x, test_set_y))

print('\nLinearRegression:')
from sklearn.linear_model import LinearRegression
Regression_(LinearRegression())

print('\nDecisionTreeRegressor:')
from sklearn.tree import DecisionTreeRegressor
Regression_(DecisionTreeRegressor())

print('\nRandomForestRegressor:')
from sklearn.ensemble import RandomForestRegressor
Regression_(RandomForestRegressor(n_estimators = 5))

print('\nExtraTreesRegressor:')
from sklearn.ensemble import ExtraTreesRegressor
Regression_(ExtraTreesRegressor(n_estimators = 5))

print('\nAdaBoostRegressor:')
from sklearn.ensemble import AdaBoostRegressor
Regression_(AdaBoostRegressor(n_estimators = 5))

print('\nGradientBoostingRegressor:')
from sklearn.ensemble import GradientBoostingRegressor
Regression_(GradientBoostingRegressor(n_estimators = 5))
