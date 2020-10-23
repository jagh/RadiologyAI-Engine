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

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier



from radiomics import featureextractor, firstorder, shape
import six
import csv


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


#######################################################################
## Feature extraction with pyradiomics
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
## Machine Learning Classifier

## Set file to write the ML outputs
ml_folder = os.path.join(testbed, "mosmeddata/machine_learning")
# Utils().mkdir(ml_folder)

## Define location to write the model
model_path = str(ml_folder+'/models/')
Utils().mkdir(model_path)

## Define location to write the one-hot transformation fot the labels
oh_path = str(ml_folder+'/models/')

## ML hyperparameters
## Set oh_flat True to use one_hot labels transformation
oh_flat = False
## Set the number of different dataset splits
n_splits = 2

lr_params = dict({'solver': 'lbfgs', 'max_iter': 500, 'random_state':2,
                                    'multi_class':'multinomial', 'n_jobs': 8})

rf_params = dict({'n_estimators': 1000, 'n_jobs': 8})
                                    # 'criterion': 'gini', 'max_features': 'sqrt',
                                    # 'max_depth': 8, 'min_samples_split': 5,
                                    # 'random_state': 2,

gb_params = dict({'learning_rate': 0.01,'n_estimators': 400, 'max_leaf_nodes': 4,
                                    'max_depth': 8, 'min_samples_split': 5,
                                    'random_state': 2, 'n_iter_no_change': 5, 'tol':0.01 })

## Set the machine learning classifiers to train
classifiers = [#MultinomialNB(),
                LogisticRegression(),
                RandomForestClassifier(),
                # GradientBoostingClassifier()
                ]

## Read the dataset for training the model
print("---"*20)
feature_extration_file = os.path.join(testbed, "mosmeddata/radiomics_features/radiomics_features.csv")
data = pd.read_csv(feature_extration_file, sep=',')
X_data = data.values[:,2:]
y_data = data.values[:,1]
print("X_train: {} || y_train: {} ".format(str(X_data.shape), str(y_data.shape)))
print("---"*20)



## Launcher a machine laerning finetune
mlc = MLClassifier()
train_scores, valid_scores = mlc.gridSearch(classifiers, X_data, y_data, oh_flat, n_splits, model_path)


## Plot the learning curves by model
mlc.plot_learning_curves(train_scores, valid_scores, n_splits)
