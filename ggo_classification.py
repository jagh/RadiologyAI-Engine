"""
Disease severity classification of lung tissue abnormalities
"""

import sys, os
import glob
import re

import pandas  as pd
import numpy   as np
import nibabel as nib
import matplotlib.pyplot as plt

import SimpleITK as sitk
from third_party.lungmask import mask
from third_party.lungmask import resunet
from engine.utils import Utils
from engine.ml_classifier import MLClassifier

from radiomics import featureextractor, firstorder, shape
import six
import csv

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import plot_confusion_matrix



def lung_lobes_segmentation(input_ct):
    """
    CT lung lobes segmentation using UNet
    """
    model = mask.get_model('unet','LTRCLobes')
    ct_segmentation = mask.apply(input_ct, model, batch_size=100)

    ### Write segmentation
    result_out = sitk.GetImageFromArray(ct_segmentation)
    result_out.CopyInformation(input_ct)
    # result_out = np.rot90(np.array(result_out)) ## modifyed the orientation

    return result_out


def loop_segmentation(input_folder, output_folder):
    """
    Multiple CT lung lobes segmentation using UNet from an input folder
    """

    for input_path in input_folder:

        ct_name = input_path.split(os.path.sep)[-1]
        ct_dcm_format = str(ct_name.split('.nii.gz')[0] + "-lung_lobes.nii.gz")

        input_ct = sitk.ReadImage(input_path)
        result_out = lung_lobes_segmentation(input_ct)

        Utils().mkdir(output_folder)
        sitk.WriteImage(result_out, str(output_folder+"/"+ct_dcm_format))
        print("CT segmentation file: {}".format(str(output_folder+"/"+ct_dcm_format)))


def feature_extraction(ct_image_path, ct_mask_path, study_name, label):
    """
    Using pyradiomics to extract first order and shape features
    """
    ct_image = sitk.ReadImage(ct_image_path)
    ct_mask = sitk.ReadImage(ct_mask_path)
    image_feature_list = []

    image_feature_list.append(study_name)
    image_feature_list.append(label)


    ## Get the First Order features
    firstOrderFeatures = firstorder.RadiomicsFirstOrder(ct_image, ct_mask)
    extractor_firstOrder = firstOrderFeatures.execute()
    for (key, val) in six.iteritems(firstOrderFeatures.featureValues):
        # print("\t%s: %s" % (key, val))
        image_feature_list.append(val)


    ## Get Shape Features in 3D
    shapeFeatures3D = shape.RadiomicsShape(ct_image, ct_mask)
    extractor_shape = shapeFeatures3D.execute()
    for (key, val) in six.iteritems(shapeFeatures3D.featureValues):
        # print("\t%s: %s" % (key, val))
        image_feature_list.append(val)

    return image_feature_list



#######################################################################
## Workflow Launcher settings
#######################################################################

#######################################################################
## CT lung lobes segmentation
testbed = "testbed/"
# studies_folder = glob.glob("../09_CT_Datasets/dataset_covid-1110_ct-scans/COVID19_1110/studies/*")
# segmentation_folder = os.path.join(testbed, "mosmeddata/lobes_segms/")
#
# for std_path in studies_folder:
#     std_folder = glob.glob(str(std_path + "/*"))
#     loop_segmentation(std_folder, segmentation_folder)
#
#
# #######################################################################
# ## Feature extraction with pyradiomics
# metadata_path = os.path.join(testbed, "mosmeddata/metadata-covid19_1110.csv")
# metadata = pd.read_csv(metadata_path, sep=',')
#
# ## Set file to write features
# radiomics_folder = os.path.join(testbed, "mosmeddata/radiomics_features")
# Utils().mkdir(radiomics_folder)
# filename = os.path.join(radiomics_folder, "radiomics_features.csv")
# f = open(filename, 'w+')
#
# for row in metadata.iterrows():
#     ## Setting files path
#     ct_file_name = row[1]["study_file"].split(os.path.sep)[-1]
#     study_name = ct_file_name.split(".nii.gz")[0]
#     seg_file_name = str(study_name + "-lung_lobes.nii.gz")
#     label =  row[1]["category"]
#     ct_segmentation_path = os.path.join(testbed, "mosmeddata/lobes_segms/", seg_file_name)
#     ct_source_path = str("../09_CT_Datasets/dataset_covid-1110_ct-scans/COVID19_1110/"+row[1]["study_file"])
#
#     ## Feature extraction by image
#     image_feature_list = feature_extraction(ct_source_path, ct_segmentation_path, study_name, label)
#
#     ## writing features by image
#     csvw = csv.writer(f)
#     csvw.writerow(image_feature_list)
#
# print("metadata: ", metadata.shape)


#######################################################################
## Machine Learning Classifier Pipeline
## Set file to write the ML outputs
ml_folder = os.path.join(testbed, "mosmeddata/machine_learning")
# Utils().mkdir(ml_folder)

##############################################################################
## Stage 1: Training and evaluation data split

## Read the dataset for training the model
print("---"*20)
feature_extration_file = os.path.join(testbed, "mosmeddata/radiomics_features/radiomics_features.csv")
data = pd.read_csv(feature_extration_file, sep=',')
X_data = data.values[:,2:]
y_data = data.values[:,1]
print("X_data: {} || y_data: {} ".format(str(X_data.shape), str(y_data.shape)))


# ## One-Hot encoding for the labels
# onehotencoder = OneHotEncoder()
# #reshape the 1-D country array to 2-D as fit_transform expects 2-D and finally fit the object
# y_data = onehotencoder.fit_transform(y_data.reshape(-1,1)).toarray()
# # print("y_1hot: {} ".format(y_1hot))
# # print("y_1hot: {} ".format(str(y_1hot.shape)))
# print("y_data: {} ".format(str(y_data.shape)))


eval_split = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=0)
for train_index, test_index in eval_split.split(X_data, y_data):
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    #print("train_index: {} || test_index: {} ".format(str(train_index.shape), str(test_index.shape) ))
print("X_train: {} || y_train: {} ".format(str(X_train.shape), str(y_train.shape)))
print("X_t: {} || y_test: {} ".format(str(X_test.shape), str(y_test.shape) ))
print("--"*20)

## Define location to write the split dataset
dataset_path = str(ml_folder+'/dataset/')
Utils().mkdir(dataset_path)

## Write the metadata file in txt
np.save(str(dataset_path+"/X_train_baseline"), X_train)
np.save(str(dataset_path+"/y_train_baseline"), y_train)
np.save(str(dataset_path+"/X_test_baseline"), X_test)
np.save(str(dataset_path+"/y_test_baseline"), y_test)



#######################################################################
## Stage 2: ML Training and Grid Search

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
                # LogisticRegression(**lr_params),
                RandomForestClassifier(**rf_params),
                GradientBoostingClassifier(**gb_params)
                ]

## Read the dataset for training the model
print("---"*20)
X_train = np.load(str(dataset_path+"/X_train_baseline.npy"), allow_pickle=True)
y_train = np.load(str(dataset_path+"/y_train_baseline.npy"), allow_pickle=True)
print("X_train: {} || y_train: {} ".format(str(X_train.shape), str(y_train.shape)))
print("---"*20)

# ## Launcher a machine laerning finetune
mlc = MLClassifier()
train_scores, valid_scores = mlc.gridSearch(classifiers, X_train, y_train, oh_flat, n_splits, model_path)

## Plot the learning curves by model
mlc.plot_learning_curves(train_scores, valid_scores, n_splits)



#######################################################################
## Stage 3: ML evaluation

## Select the model to evaluate
model_name = 'RandomForestClassifier'
                #'RandomForestClassifier'
                #'GradientBoostingClassifier'

## Read the dataset for training the model
print("---"*20)
X_test = np.load(str(dataset_path+"/X_test_baseline.npy"), allow_pickle=True)
y_test = np.load(str(dataset_path+"/y_test_baseline.npy"), allow_pickle=True)
print("X_train: {} || y_train: {} ".format(str(X_test.shape), str(y_test.shape)))
print("---"*20)

## ML evaluation
page_clf, test_score = mlc.model_evaluation(model_path, model_name, X_test, y_test, oh_flat)

## Plot the the confusion matrix by model selected
plot_confusion_matrix(page_clf, X_test, y_test, xticks_rotation=15  )
plt.title(str(model_name+" || F1-score: "+ str(test_score)))
plt.show()
