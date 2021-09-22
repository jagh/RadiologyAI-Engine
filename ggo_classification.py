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

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



#######################################################################
## Launcher settings to classify a ground-glass opacities score
#######################################################################
# ## Dataset path definitions
# studies_folder = glob.glob("/data/01_UB/CT_Datasets/dataset_covid-1110_ct-scans/COVID19_1110/studies/*")
# testbed = "testbed/"
#
# bilung_segmentation_folder = os.path.join(testbed, "mosmeddata/segmentations_bi-lung/")
# lobes_segmentation_folder = os.path.join(testbed, "mosmeddata/segmentations_lobes/")
#
#
# #######################################################################
# ## Stage-1: bi-lung and lung lobes CT segmentation
# def auto_segmentation(studies_folder, segmentation_folder, seg_method):
#     """ Formatting the folder studies for each GGO categories """
#     for std_path in studies_folder:
#         input_std = glob.glob(str(std_path + "/*"))
#         folder_name = std_path.split(os.path.sep)[-1]
#         output_std = os.path.join(segmentation_folder, folder_name)
#         Utils().mkdir(output_std)
#
#         ## Launch of automatic CT segmentation
#         LungSegmentations().folder_segmentations(input_std, output_std, seg_method, 5)
#
# # auto_segmentation(studies_folder, bilung_segmentation_folder, 'bi-lung')
# # auto_segmentation(studies_folder, lobes_segmentation_folder, 'lobes')



# #######################################################################
# ## Stage-2: Feature extraction with pyradiomics
# ## https://www.radiomics.io/pyradiomics.html
# studies_path = "/data/01_UB/CT_Datasets/dataset_covid-1110_ct-scans/COVID19_1110/studies/"
# metadata_file = os.path.join(testbed, "mosmeddata/metadata-covid19_1110.csv")
# metadata = pd.read_csv(metadata_file, sep=',')
# print("metadata: ", metadata.shape)
#
# ## Crete new folder for feature extraction
# radiomics_folder = os.path.join(testbed, "mosmeddata/radiomics_features/")
# Utils().mkdir(radiomics_folder)
#
# ## Loop to extract features for an especific segmentation label
# ## bi-lung: {l1: right, l2: left}
# ## lobes: {l1: l_upper, l2: l_lower, l3: r_upper , l4: r_middle, l5: r_lower}
# for lobes_area in range(5):
#
#     ## Set file name to write a features vector per case
#     lobes_area=str(lobes_area+1)
#     filename = str(radiomics_folder+"/radiomics_features-"+lobes_area+".csv")
#     features_file = open(filename, 'w+')
#
#     for row in metadata.iterrows():
#         ## Getting the GGO label
#         label =  row[1]["category"]
#
#         ## Locating the ct image file
#         ct_image_name = row[1]["study_file"].split(os.path.sep)[-1]
#         ct_image_path = os.path.join(studies_path, label, ct_image_name)
#         ct_case_id = ct_image_name.split(".nii.gz")[0]
#
#         # ## Locating the bi-lung segmentation file
#         # bilung_segmentation_name = str(ct_case_id + "-bi-lung.nii.gz")
#         # bilung_segmentation_path = os.path.join(bilung_segmentation_folder, label, bilung_segmentation_name)
#         #
#         # ##Feature extraction by image
#         # re = RadiomicsExtractor(lobes_area)
#         # image_feature_list = re.feature_extractor(ct_image_path, bilung_segmentation_path, ct_case_id, label)
#
#
#         ## Locating the lobes segmentation file
#         lobes_segmentation_name = str(ct_case_id + "-lobes.nii.gz")
#         lobes_segmentation_path = os.path.join(lobes_segmentation_folder, label, lobes_segmentation_name)
#
#         ## Feature extraction by image
#         re = RadiomicsExtractor(lobes_area)
#         image_feature_list = re.feature_extractor(ct_image_path, lobes_segmentation_path, ct_case_id, label)
#
#         ## writing features by image
#         csvw = csv.writer(features_file)
#         csvw.writerow(image_feature_list)



#######################################################################
## Stage 3: Machine learning pipeline


# ##---------------------------------------
# ## Set full-lungs files path
# radiomics_folder = os.path.join(testbed, "mosmeddata/radiomics_features/")
# left_lung = os.path.join(radiomics_folder, "radiomics_features-full_lung.csv")
# X_data, y_data = load_features(left_lung)


##---------------------------------------
## Set bi-lungs files path
# radiomics_folder = os.path.join(testbed, "mosmeddata/radiomics_features/")
# left_lung = os.path.join(radiomics_folder, "radiomics_features-left_lung.csv")
# right_lung = os.path.join(radiomics_folder, "radiomics_features-right_lung.csv")
#
# X_left_lung, y_left_lung = load_features(left_lung)
# X_right_lung, y_right_lung = load_features(right_lung)
#
# X_data = np.concatenate((X_left_lung, X_right_lung), axis=1)
# y_data = y_left_lung
# print("X_data: {} || y_data: {} ".format(str(X_data.shape), str(y_data.shape)))


# ##---------------------------------------
# ## Set lungs lobes files path
# radiomics_folder = os.path.join(testbed, "mosmeddata/radiomics_features/")
# left_lower = os.path.join(radiomics_folder, "radiomics_features-left_lower.csv")
# left_upper = os.path.join(radiomics_folder, "radiomics_features-left_upper.csv")
# right_lower = os.path.join(radiomics_folder, "radiomics_features-right_lower.csv")
# right_middle = os.path.join(radiomics_folder, "radiomics_features-right_middle.csv")
# right_upper = os.path.join(radiomics_folder, "radiomics_features-right_upper.csv")

# X_left_lower, y_left_lower = load_features(left_lower)
# X_left_upper, y_left_upper = load_features(left_upper)
# X_right_lower, y_right_lower = load_features(right_lower)
# X_right_middle, y_right_middle = load_features(right_middle)
# X_right_upper, y_right_upper = load_features(right_upper)

# X_data = np.concatenate((X_left_lower, X_left_upper,
#                         X_right_lower, X_right_middle, X_right_upper), axis=1)
# y_data = y_left_lower



##############################################################################




##------------------------------------------------------------------------------
##------------------------------------------------------------------------------
## Step-1: From multiple files define features, labels and spliting the dataset
def load_features(file_path):
    """Read features and labels per file"""
    data = pd.read_csv(file_path, sep=',', header=0)
    ## Set features and labels, discard the two cases for a GGO 'CT-4'
    X_data = data.values[:,2:].astype(np.float).astype("Int32")  #[:1107,2:]
    y_data = data.values[:,1]   #[:1107,1]
    y_data=y_data.astype('int')
    print("X_data: {} || y_data: {} ".format(str(X_data.shape), str(y_data.shape)))
    return X_data, y_data


## Set leion segmentation file
testbed = "testbed/"
# # radiomics_folder = os.path.join(testbed, "2D-MulticlassLesionSegmentation/radiomics_features/")
radiomics_folder = os.path.join(testbed, "MULTICLASS/radiomics_features/")
lesion_features_file_path = os.path.join(radiomics_folder, "lesion_features.csv")
X_data, y_data = load_features(lesion_features_file_path)



## Create a ML folder and splitting the dataset
# ml_folder = os.path.join(testbed, "2D-MulticlassLesionSegmentation/machine_learning")
ml_folder = os.path.join(testbed, "MULTICLASS/machine_learning/")
Utils().mkdir(ml_folder)
MLClassifier().splitting(X_data, y_data, ml_folder)




##------------------------------------------------------------------------------
##------------------------------------------------------------------------------
## Step-2: ML Training and Grid Search
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


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

## Set the machine learning classifiers to train
classifiers = [
                LogisticRegression(**lr_params),
                RandomForestClassifier(**rf_params),
                GradientBoostingClassifier(**gb_params),
                ]

## Read the dataset for training the model
print("---"*20)
X_train = np.load(str(ml_folder+'/dataset/'+"/X_train_baseline.npy"), allow_pickle=True)
y_train = np.load(str(ml_folder+'/dataset/'+"/y_train_baseline.npy"), allow_pickle=True)
print("X_train: {} || y_train: {} ".format(str(X_train.shape), str(y_train.shape)))

## Launcher a machine laerning finetune
mlc = MLClassifier()
train_scores, valid_scores = mlc.gridSearch(classifiers, X_train, y_train,
                                                    oh_flat, n_splits, model_path)

## Plot the learning curves by model
mlc.plot_learning_curves(train_scores, valid_scores, n_splits)



#
# ##------------------------------------------------------------------------------
# ##------------------------------------------------------------------------------
# # Step 3: Model evaluation
# from sklearn.metrics import plot_confusion_matrix
#
# ## Select the model to evaluate
# model_name = 'RandomForestClassifier'
#                 #'LogisticRegression'
#                 #'RandomForestClassifier'
#                 #'GradientBoostingClassifier'
#
# ## Read the dataset for training the model
# testbed = "testbed/"
# # ml_folder = os.path.join(testbed, "2D-MulticlassLesionSegmentation/machine_learning")
# ml_folder = os.path.join(testbed, "MULTICLASS/machine_learning/")
# model_path = str(ml_folder+'/models/')
#
# ## Use oh_flat to encode labels as one-hot for RandomForestClassifier
# oh_flat = True
#
# X_test = np.load(str(ml_folder+'/dataset/'+"/X_test_baseline.npy"), allow_pickle=True)
# y_test = np.load(str(ml_folder+'/dataset/'+"/y_test_baseline.npy"), allow_pickle=True)
# print("---"*20)
# print("X_eval: {} || y_eval: {} ".format(str(X_test.shape), str(y_test.shape)))
#
# ## ML evaluation
# mlc = MLClassifier()
# page_clf, test_score = mlc.model_evaluation(model_path, model_name, X_test, y_test, oh_flat)
#
# ## Plot the the confusion matrix by model selected
# # labels_name = ['CT-0', 'CT-1', 'CT-2', 'CT-3']
# # labels_name = ['Healthy', 'Non-healthy']
# labels_name = ['Intubated', 'Non-Intubated']
# plot_confusion_matrix(page_clf, X_test, y_test,
#                             display_labels=labels_name,
#                             cmap=plt.cm.Blues,
#                             # normalize='true'
#                             ) #, xticks_rotation=15)
# plt.title(str(model_name+" || F1-score: "+ str(test_score)))
# plt.show()
#






###################################################################
## TO DO list
## Tk-1: ML Model Interpretability — LIME || https://medium.com/analytics-vidhya/model-interpretability-lime-part-2-53c0f5e76b6a
## Tk-2: Interpretability of Machine Learning models || https://medium.com/analytics-vidhya/interpretability-of-machine-learning-models-9787cf8a3789


## Tk-1: Check the actual process in ML Stage -> Done
## Tk-2: Add the 1_hot transformation for the labels -> Done
## Tk-3: Merge radiomics features by segmentations areas
## Tk-4: Compute the HU by segmentation areas
## Ref -> https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
## Ref-2 -> https://vincentblog.xyz/posts/medical-images-in-python-computed-tomography
