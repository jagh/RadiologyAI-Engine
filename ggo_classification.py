"""
Disease severity classification of lung tissue abnormalities using the MosMedata.
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




#######################################################################
## Launcher settings to classify a ground-glass opacities score
#######################################################################

## Dataset path definitions
studies_folder = glob.glob("/data/01_UB/CT_Datasets/dataset_covid-1110_ct-scans/COVID19_1110/studies/*")
testbed = "testbed/"
bilung_segmentation_folder = os.path.join(testbed, "mosmeddata/segmentations_bi-lung/")
lobes_segmentation_folder = os.path.join(testbed, "mosmeddata/segmentations_lobes/")


#######################################################################
## Stage-1: bi-lung and lung lobes CT segmentation
def auto_segmentation(studies_folder, segmentation_folder, seg_method):
    """ Formatting the folder studies for each GGO categories """
    for std_path in studies_folder:
        input_std = glob.glob(str(std_path + "/*"))
        folder_name = std_path.split(os.path.sep)[-1]
        output_std = os.path.join(segmentation_folder, folder_name)
        Utils().mkdir(output_std)

        ## Launch of automatic CT segmentation
        LungSegmentations().folder_segmentations(input_std, output_std, seg_method, 5)

auto_segmentation(studies_folder, bilung_segmentation_folder, 'bi-lung')
auto_segmentation(studies_folder, lobes_segmentation_folder, 'lobes')



#######################################################################
## Stage-2: Feature extraction with pyradiomics
## https://www.radiomics.io/pyradiomics.html
studies_path = "/data/01_UB/CT_Datasets/dataset_covid-1110_ct-scans/COVID19_1110/studies/"
metadata_file = os.path.join(testbed, "mosmeddata/metadata-covid19_1110.csv")
metadata = pd.read_csv(metadata_file, sep=',')
print("metadata: ", metadata.shape)

## Crete new folder for feature extraction
radiomics_folder = os.path.join(testbed, "mosmeddata/radiomics_features")
Utils().mkdir(radiomics_folder)

## Set file name to write a features vector per case
filename = os.path.join(radiomics_folder, "radiomics_features.csv")
features_file = open(filename, 'w+')

for row in metadata.iterrows():
    ## Getting the GGO label
    label =  row[1]["category"]

    ## Locating the ct image file
    ct_image_name = row[1]["study_file"].split(os.path.sep)[-1]
    ct_image_path = os.path.join(studies_path, label, ct_image_name)
    ct_case_id = ct_image_name.split(".nii.gz")[0]

    ## Locating the bi-lung segmentation file
    bilung_segmentation_name = str(ct_case_id + "-bi-lung.nii.gz")
    bilung_segmentation_path = os.path.join(bilung_segmentation_folder, label, bilung_segmentation_name)

    ## Feature extraction by image
    re = RadiomicsExtractor()
    image_feature_list = re.feature_extractor(ct_image_path, bilung_segmentation_path, ct_case_id, label)

    ## writing features by image
    csvw = csv.writer(features_file)
    csvw.writerow(image_feature_list)



#######################################################################
## Stage 3: ML Training and Grid Search

## Read the dataset for training the model
feature_extration_file = os.path.join(testbed, "mosmeddata/radiomics_features/radiomics_features.csv")
ml_folder = os.path.join(testbed, "mosmeddata/machine_learning")

data = pd.read_csv(feature_extration_file, sep=',')
X_data = data.values[:,2:]
y_data = data.values[:,1]
print("---"*20)
print("X_data: {} || y_data: {} ".format(str(X_data.shape), str(y_data.shape)))

## Create a ML folder and splitting the dataset
MLClassifier().splitting(X_data, y_data, ml_folder)


#######################################################################
## ML Training and Grid Search
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

## Define location to write the model
model_path = str(ml_folder+'/models/')
Utils().mkdir(model_path)

## Define location to write the one-hot transformation fot the labels
oh_path = str(ml_folder+'/models/')

## ML hyperparameters
## Set oh_flat True to use one_hot labels transformation
oh_flat = False

## Set the number of different dataset splits
n_splits = 5

## 'newton-cg', 'lbfgs', 'liblinear'
lr_params = dict({'solver': 'newton-cg', 'max_iter': 1000, 'random_state':2,
                                    'multi_class':'multinomial', 'n_jobs': 8})

## https://www.analyticsvidhya.com/blog/2020/03/beginners-guide-random-forest-hyperparameter-tuning/
rf_params = dict({'bootstrap': True, 'class_weight': None,
                    'criterion': 'gini', 'n_estimators': 2000,
                    'max_features': 'auto', 'max_depth': 5,
                    'min_samples_leaf': 1, 'min_samples_split': 2,
                    'random_state': 2, 'n_jobs': 8})

## https://medium.com/all-things-ai/in-depth-parameter-tuning-for-gradient-boosting-3363992e9bae
gb_params = dict({'criterion': 'friedman_mse', 'init': None,
                    'learning_rate': 0.1, 'loss': 'deviance', 'max_depth': 3,
                    'min_samples_leaf': 0.50, 'min_samples_split': 2,
                    'n_estimators': 200, 'random_state': None,
                    'max_features': None, 'max_leaf_nodes': None,
                    'n_iter_no_change': None, 'tol':0.01 })



## Set the machine learning classifiers to train
classifiers = [#MultinomialNB(),
                LogisticRegression(**lr_params),
                RandomForestClassifier(**rf_params),
                GradientBoostingClassifier(**gb_params)
                ]

## Read the dataset for training the model
print("---"*20)
X_train = np.load(str(ml_folder+'/dataset/'+"/X_train_baseline.npy"), allow_pickle=True)
y_train = np.load(str(ml_folder+'/dataset/'+"/y_train_baseline.npy"), allow_pickle=True)
print("X_train: {} || y_train: {} ".format(str(X_train.shape), str(y_train.shape)))
print("---"*20)

## Launcher a machine laerning finetune
mlc = MLClassifier()
train_scores, valid_scores = mlc.gridSearch(classifiers, X_train, y_train, oh_flat, n_splits, model_path)

## Plot the learning curves by model
mlc.plot_learning_curves(train_scores, valid_scores, n_splits)


# #######################################################################
# ## Stage 3: ML evaluation
# from sklearn.metrics import plot_confusion_matrix
#
# ## Select the model to evaluate
# model_name = 'GradientBoostingClassifier'
#                 #'LogisticRegression'
#                 #'RandomForestClassifier'
#
# ## Read the dataset for training the model
# X_test = np.load(str(ml_folder+'/dataset/'+"/X_test_baseline.npy"), allow_pickle=True)
# y_test = np.load(str(ml_folder+'/dataset/'+"/y_test_baseline.npy"), allow_pickle=True)
# print("X_train: {} || y_train: {} ".format(str(X_test.shape), str(y_test.shape)))
# print("---"*20)
#
# ## ML evaluation
# mlc = MLClassifier()
# page_clf, test_score = mlc.model_evaluation(model_path, model_name, X_test, y_test, oh_flat)
#
# ## Plot the the confusion matrix by model selected
# plot_confusion_matrix(page_clf, X_test, y_test, xticks_rotation=15)
# plt.title(str(model_name+" || F1-score: "+ str(test_score)))
# plt.show()


###################################################################
## TO DO
## Compute the HU -> https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
