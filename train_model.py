import prepare_dataset

# Common imports
import pandas as pd
import numpy as np
import scipy as sp
import collections
import os
import sys
import math
from math import pi
from math import sqrt
import csv
import urllib
import pickle
import random
import statsmodels.api as sm
from patsy import dmatrices

# Math and descriptive stats
from math import sqrt
from scipy import stats

# Sci-kit Learn modules for machine learning
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, VotingClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding
from sklearn.svm import SVR, LinearSVC, SVC, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA, KernelPCA

# Boosting libraries
import lightgbm as lgb
import xgboost

# Deep Learning modules
from keras.layers import Input, Dense, Dropout, Flatten, Embedding, merge, Activation
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.utils import np_utils, to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf

from trueskill import TrueSkill, Rating, rate_1vs1

Xtrain = np.load("Data/PrecomputedMatrices/X_train.npy")
ytrain = np.load("Data/PrecomputedMatrices/y_train.npy")
Xtrain = np.nan_to_num(Xtrain)
ytrain = np.nan_to_num(ytrain)

X_train, X_test, Y_train, Y_test = train_test_split(Xtrain, ytrain)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

log = LogisticRegression(random_state=95)
log.fit(X_train, Y_train)

knn = KNeighborsClassifier(algorithm='kd_tree', leaf_size=18, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=60, p=1,
           weights='uniform')

rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=40, max_features='log2', max_leaf_nodes=200,
            min_impurity_decrease=0, min_impurity_split=None,
            min_samples_leaf=4, min_samples_split=500,
            min_weight_fraction_leaf=0, n_estimators=1000, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

etrees = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
           max_depth=40, max_features='log2', max_leaf_nodes=200,
           min_impurity_decrease=0, min_impurity_split=None,
           min_samples_leaf=4, min_samples_split=500,
           min_weight_fraction_leaf=0, n_estimators=1000, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)

gbc = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=3,
              max_features=None, max_leaf_nodes=16,
              min_impurity_decrease=0.2, min_impurity_split=None,
              min_samples_leaf=220, min_samples_split=2,
              min_weight_fraction_leaf=0, n_estimators=150, presort='auto',
              random_state=None, subsample=1, verbose=0, warm_start=False)

lgbm = LGBMClassifier(bagging_freq=0, bagging_seed=95, boosting_type='gbdt',
        class_weight=None, colsample_bytree=1.0, feature_fraction=1.0,
        feature_fraction_seed=95, lambda_l1=0.0, lambda_l2=0.0,
        learning_rate=0.1, max_depth=12, min_child_samples=20,
        min_child_weight=0.001, min_data_in_leaf=10, min_split_gain=0,
        n_estimators=100, n_jobs=-1, num_boost_round=40, num_leaves=25,
        num_threads=4, objective=None, random_state=None, reg_alpha=0.0,
        reg_lambda=0.0, silent=True, subsample=1.0,
        subsample_for_bin=200000, subsample_freq=1)

def model_function(layer_one_neurons=184, layer_two_neurons=184, layer_three_neurons=184):
    model = Sequential()
    model.add(Dense(layer_one_neurons, input_dim=Xtrain_nn.shape[1], activation='relu'))
    model.add(Dense(layer_two_neurons, activation='relu'))
    model.add(Dense(layer_three_neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
model = KerasClassifier(build_fn=model_function, epochs=100)

# Soft Voting to get the probabilities
model_soft = VotingClassifier(estimators=[('keras', model), ('gbc', gbc), ('rfr', rfr), 
                                          ('etrees', etrees), ('knn', knn), ('lgbm', lgbm),
                                          ('log', log)], voting='soft')

model_soft.fit(X_train, Y_train)

def predictGame(team_1_vector, team_2_vector, home):
    diff = [a - b for a, b in zip(team_1_vector, team_2_vector)]
    diff.append(home)

    return model_soft.predict_proba(np.array([diff]))[0][1]
    #return calibrated_model.predict([diff])[0][1] # Depends on model(s) chosen

def loadTeamVectors(years):
    listDictionaries = []
    for year in years:
        curVectors = np.load("Data/PrecomputedMatrices/TeamVectors/" + str(year) + "TeamVectors.npy").item()
        listDictionaries.append(curVectors)
    return listDictionaries

def createPrediction():
    if os.path.exists("Data/Predictions/class_results.csv"):
        os.remove("Data/Predictions/class_results.csv")
    years = range(2014,2018)
    listDictionaries = loadTeamVectors(years)
    results = [[0 for x in range(2)] for x in range(len(sample_sub_pd.index))]
    for index, row in sample_sub_pd.iterrows():
        matchupId = row['ID']
        year = int(matchupId[0:4]) 
        teamVectors = listDictionaries[year - years[0]]
        team1Id = int(matchupId[5:9])
        team2Id = int(matchupId[10:14])
        team1Vector = teamVectors[team1Id] 
        team2Vector = teamVectors[team2Id]
        pred = predictGame(team1Vector, team2Vector, 0)
        results[index][0] = matchupId
        results[index][1] = pred
    results = pd.np.array(results)
    firstRow = [[0 for x in range(2)] for x in range(1)]
    firstRow[0][0] = 'ID'
    firstRow[0][1] = 'Pred'
    with open("Data/Predictions/class_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(firstRow)
        writer.writerows(results)

class_results = pd.read_csv('Data/Predictions/class_results.csv')
trueskill_results = pd.read_csv('Data/Predictions/trueskill_results.csv')

final_predictions = sample_sub_pd
final_predictions.Pred = (class_results.Pred * 0.8 + trueskill_pd.Pred * 0.2)

final_predictions.to_csv('NCAA_Predictions.csv', index=None)