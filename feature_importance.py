"""
Feature importace using LASSO
"""

import sys, os
import glob
import pandas as pd
import numpy as np
import csv

from engine.utils import Utils
from engine.segmentations import LungSegmentations
from engine.featureextractor import RadiomicsExtractor
from engine.ml_classifier import MLClassifier

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



##------------------------------------------------------------------------------
##------------------------------------------------------------------------------
## Step-1: From multiple files define features, labels and spliting the dataset
def load_features(file_path):
    """Read features and labels per file"""
    data = pd.read_csv(file_path, sep=',', header=0)
    ## Set features and labels, discard the two cases for a GGO 'CT-4'
    X_data = data.values[:,3:].astype(np.float).astype("Int32")  #[:1107,2:]
    y_data = data.values[:,2]   #[:1107,1]
    y_data=y_data.astype('int')
    print("X_data: {} || y_data: {} ".format(str(X_data.shape), str(y_data.shape)))
    return X_data, y_data, data.columns.values[3:]


## Set leion segmentation file
testbed = "testbed-ECR22/"


# ########################################################
# experiment_name = "LUNG"
# # experiment_filename = "lesion_features-0-Tr.csv"
# experiment_filename = "lesion_features-0-Ts.csv"


# ########################################################
# experiment_name = "01_GGO&CON"
# experiment_filename = "general_lesion_features-Tr.csv"
# experiment_filename = "general_lesion_features-Ts.csv"

# experiment_filename = "general_lesion_features-Tr-withLung-FullFeatures.csv"
# experiment_filename = "general_lesion_features-Ts-withLung-FullFeatures.csv"



# # ########################################################
experiment_name = "02_GENERAL"
# experiment_filename = "general2class-Tr-FeatureSelection.csv"
experiment_filename = "general2class-Ts-FeatureSelection.csv"


# # ########################################################
# experiment_name = "03_MULTICLASS"

## Lesion-1
# experiment_filename = "multi2class_lwl-5-Tr-FeatureSelection.csv"
# experiment_filename = "multi2class_lwl-5-Ts-FeatureSelection.csv"

## cov2radiomics
# experiment_filename = "cov2radiomics-Tr-FeatureSelection.csv"
# experiment_filename = "cov2radiomics-Ts-FeatureSelection.csv"



radiomics_folder = os.path.join(testbed, experiment_name, "radiomics_features/")
lesion_features_file_path = os.path.join(radiomics_folder, experiment_filename)
X_data, y_data, features = load_features(lesion_features_file_path)


## Create a ML folder and splitting the dataset
ml_folder = os.path.join(testbed, experiment_name, "machine_learning/")
Utils().mkdir(ml_folder)
MLClassifier().splitting(X_data, y_data, ml_folder)



##------------------------------------------------------------------------------
##------------------------------------------------------------------------------
## Step-2: ML Training and Grid Search
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import plot_confusion_matrix

## Feature importance
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

## Define location to write the model
# ml_folder = os.path.join(testbed, "2D-MulticlassLesionSegmentation/machine_learning")
# ml_folder = os.path.join(testbed, "MULTICLASS-2/machine_learning/")
model_path = str(ml_folder+'/models/')
Utils().mkdir(model_path)

## Use oh_flat to encode labels as one-hot for RandomForestClassifier
oh_flat = True

## Set the number of different dataset splits
n_splits = 5

## LogisticRegression parameters  'newton-cg', 'lbfgs'
lr_params = dict({'solver': 'newton-cg', 'max_iter': 500, 'random_state':2,
                            'multi_class':'multinomial', 'n_jobs': 8})

## https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/
rf_params = dict({'bootstrap': True, 'class_weight': None,
                    'criterion': 'gini', 'n_estimators': 2000,
                    'max_features': 'auto', 'max_depth': 5,
                    'min_samples_leaf': 1, 'min_samples_split': 2,
                    'random_state': 2, 'n_jobs': 8})

## https://medium.com/all-things-ai/in-depth-parameter-tuning-for-gradient-boosting-3363992e9bae
gb_params = dict({'criterion': 'friedman_mse', 'init': None,
                    'learning_rate': 0.1, 'loss': 'deviance', 'max_depth': 5,
                    'min_samples_leaf': 1, 'min_samples_split': 2,
                    'n_estimators': 1000, 'random_state': None,
                    'max_features': None, 'max_leaf_nodes': None,
                    'n_iter_no_change': None, 'tol':0.01})


##-----------------------------------------------------------------------------
##-----------------------------------------------------------------------------
# ## Set the machine learning classifiers to train
# classifiers = [
#                 LogisticRegression(**lr_params),
#                 RandomForestClassifier(**rf_params),
#                 GradientBoostingClassifier(**gb_params),
#                 ]
#
# ## Launcher a machine laerning finetune
# mlc = MLClassifier()
# train_scores, valid_scores = mlc.gridSearch(classifiers, X_train, y_train,
#                                                     oh_flat, n_splits, model_path)
#
# ## Plot the learning curves by model
# mlc.plot_learning_curves(train_scores, valid_scores, n_splits)





##-----------------------------------------------------------------------------
##-----------------------------------------------------------------------------
## Feature Importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso


## Read the dataset for training the model
print("---"*20)
Xtr = np.load(str(ml_folder+'/dataset/'+"/X_train_baseline.npy"), allow_pickle=True)
ytr = np.load(str(ml_folder+'/dataset/'+"/y_train_baseline.npy"), allow_pickle=True)
print("X_train: {} || y_train: {} ".format(str(Xtr.shape), str(ytr.shape)))


## Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(Xtr, ytr, test_size=0.33, random_state=42)

pipeline = Pipeline([
                     ('scaler',StandardScaler()),
                     ('model',Lasso())
                    ])


search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.1,10,0.1)},
                      cv = 5, scoring="neg_mean_squared_error",verbose=3
                      )

search.fit(X_train,y_train)
search.best_params_
# print("++ search.best_params_: ", search.best_params_)

## Get the values of the coefficients of Lasso regression.
coefficients = search.best_estimator_.named_steps['model'].coef_
importance = np.abs(coefficients)

## The features that survived the Lasso regression are:
print("++ importance > 0]", np.array(features)[importance > 0])


## While the 3 discarded features are
print("++ importance > 0]", np.array(features)[importance == 0])
