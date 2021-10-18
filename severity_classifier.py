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



from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


import sklearn.metrics as metrics




##------------------------------------------------------------------------------
##------------------------------------------------------------------------------
## Step-1: From multiple files define features, labels and spliting the dataset
def load_features(file_path):
    """Read features and labels per file"""
    data = pd.read_csv(file_path, sep=',', header=0)
    ## Set features and labels, discard the two cases for a GGO 'CT-4'
    # X_data = data.values[:,3:].astype(np.float).astype("Int32")  #[:1107,2:]
    X_data = data.values[:,3:].astype(np.float)  #[:1107,2:]
    y_data = data.values[:,2]   #[:1107,1]
    y_data=y_data.astype('int')
    print("X_data: {} || y_data: {} ".format(str(X_data.shape), str(y_data.shape)))
    return X_data, y_data



def load_features_index(file_path):
    """Read features and labels per file"""
    data = pd.read_csv(file_path, sep=',', header=0)
    ## Set features and labels, discard the two cases for a GGO 'CT-4'
    # X_data = data.values[:,3:].astype(np.float).astype("Int32")  #[:1107,2:]
    X_data = data.values[:,3:].astype(np.float)  #[:1107,2:]
    y_data = data.values[:,2]   #[:1107,1]
    y_data = y_data.astype('int')
    X_index = data.values[:,1].astype('str')
    print("X_data: {} || y_data: {} ".format(str(X_data.shape), str(y_data.shape)))
    return X_data, y_data, X_index


## Set leion segmentation file
testbed = "testbed-ECR22/"


# ########################################################
# experiment_name = "GENERAL"
# # experiment_filename = "general_lesion_features-Tr.csv"
# experiment_filename = "general_lesion_lung_features-Tr.csv"
# # test_lesion_features_file_path = os.path.join(radiomics_folder, "general_lesion_features-Ts.csv")


# ########################################################
# experiment_name = "01_GGO&CON"
# # experiment_filename = "general_lesion_features-Tr.csv"
# experiment_filename = "general_lesion_features-Ts.csv"


########################################################
experiment_name = "02_GENERAL"
# ## Full Features
# experiment_filename = "general2class-Tr-AllSlices.csv"

# ## Features selected
experiment_filename = "general2class-Tr-FeatureSelection-2.csv"


# ########################################################
# experiment_name = "03_MULTICLASS"
# ## Full Features
# # experiment_filename = "multi2class_lwl-1-Tr-AllSlices.csv"
#
# ## Features selected by each lesion
# experiment_filename = "multi2class_lwl-5-Tr-FeatureSelection.csv"



# #####################################################################
# ## cov2radiomics
# experiment_filename = "cov2radiomics-Tr-AllSlices.csv"
#
# experiment_filename = "cov2radiomics-Ts-FeatureSelection.csv"



radiomics_folder = os.path.join(testbed, experiment_name, "radiomics_features/")
lesion_features_file_path = os.path.join(radiomics_folder, experiment_filename)
X_data, y_data = load_features(lesion_features_file_path)


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
##Set the machine learning classifiers to train
# classifiers = [
#                 LogisticRegression(**lr_params),
#                 RandomForestClassifier(**rf_params),
#                 # GradientBoostingClassifier(**gb_params),
#                 ]
#
# ## Read the dataset for training the model
# print("---"*20)
# X_train = np.load(str(ml_folder+'/dataset/'+"/X_train_baseline.npy"), allow_pickle=True)
# y_train = np.load(str(ml_folder+'/dataset/'+"/y_train_baseline.npy"), allow_pickle=True)
# print("X_train: {} || y_train: {} ".format(str(X_train.shape), str(y_train.shape)))
#
# ## Launcher a machine laerning finetune
# mlc = MLClassifier()
# train_scores, valid_scores = mlc.gridSearch(classifiers, X_train, y_train,
#                                                     oh_flat, n_splits, model_path)
#
# ## Plot the learning curves by model
# mlc.plot_learning_curves(train_scores, valid_scores, n_splits)





# ##------------------------------------------------------------------------------
# ##------------------------------------------------------------------------------
# # Step 3: Model evaluation
# ########################################################
#
# ## Test file
# ## Only One Lsion Full Features
# # experiment_filename = "multi2class_lwl-1-Tr-AllSlices.csv"
# #
# ## General Lesion Full Features
# # test_lesion_features_file_path = os.path.join(radiomics_folder, "general2class-Ts-AllSlices.csv")
# #
# # ## Cov2Radiomics Full Features
# # test_lesion_features_file_path = os.path.join(radiomics_folder, "cov2radiomics-Ts-AllSlices.csv")
#
#
# # ## Features selected by each lesion
# # test_lesion_features_file_path = os.path.join(radiomics_folder, "multi2class_lwl-5-Tr-FeatureSelection.csv")
# #
# ## General Lesion Full Features
# test_lesion_features_file_path = os.path.join(radiomics_folder, "general2class-Ts-FeatureSelection-2.csv")
# #
# # ## Cov2Radiomics Full Features
# # test_lesion_features_file_path = os.path.join(radiomics_folder, "cov2radiomics-Ts-FeatureSelection.csv")
#
#
# # Select the model to evaluate
# model_name = 'LogisticRegression'
#                 #'LogisticRegression'
#                 #'RandomForestClassifier'
#                 #'GradientBoostingClassifier'
#
# ## Read the dataset for training the model
# ml_folder = os.path.join(testbed, experiment_name, "machine_learning/")
# model_path = str(ml_folder+'/models/')
#
# ## Use oh_flat to encode labels as one-hot for RandomForestClassifier
# oh_flat = True
#
#
# ##---------------------------------------------------------------------
# ## Load dataset
# X_test, y_test = load_features(test_lesion_features_file_path)
#
# # X_test = np.load(str(ml_folder+'/dataset/'+"/X_test_baseline.npy"), allow_pickle=True)
# # y_test = np.load(str(ml_folder+'/dataset/'+"/y_test_baseline.npy"), allow_pickle=True)
#
#
#
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
# labels_name = ['Non-Intubated', 'Intubated']
# plot_confusion_matrix(page_clf, X_test, y_test,
#                             display_labels=labels_name,
#                             cmap=plt.cm.Blues,
#                             # normalize='true'
#                             ) #, xticks_rotation=15)
# plt.title(str(model_name+" || F1-score: "+ str(test_score)))
# plt.show()
#
#
# ## ROC curves
# def plot_roc(model, X_test, y_test, model_name):
#     # calculate the fpr and tpr for all thresholds of the classification
#     probabilities = model.predict_proba(np.array(X_test))
#     predictions = probabilities[:, 1]
#     fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
#     roc_auc = metrics.auc(fpr, tpr)
#
#     plt.title(str(model_name + "") )
#     plt.plot(fpr, tpr, 'g', label='AUC = %0.2f' % roc_auc)
#     plt.legend(loc='lower right')
#     plt.plot([0, 1], [0, 1], 'r--')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.show()
#
# plot_roc(page_clf, X_test, y_test, model_name)






##------------------------------------------------------------------------------
##------------------------------------------------------------------------------
# Step 4: Model inference for patient assessment
########################################################

## Test file
## Only One Lsion Full Features
# experiment_filename = "multi2class_lwl-1-Tr-AllSlices.csv"
#
## General Lesion Full Features
# test_lesion_features_file_path = os.path.join(radiomics_folder, "general2class-Ts-AllSlices.csv")
#
# ## Cov2Radiomics Full Features
# test_lesion_features_file_path = os.path.join(radiomics_folder, "cov2radiomics-Ts-AllSlices.csv")


# ## Features selected by each lesion
# test_lesion_features_file_path = os.path.join(radiomics_folder, "multi2class_lwl-5-Tr-FeatureSelection.csv")
#
## General Lesion Full Features
# experiment_name = "02_GENERAL"
# test_lesion_features_file_path = os.path.join(radiomics_folder, "general2class-Ts-FeatureSelection-2.csv")
test_lesion_features_file_path = os.path.join(radiomics_folder, "general2class-Tr-FeatureSelection-2.csv")
#
# ## Cov2Radiomics Full Features
# test_lesion_features_file_path = os.path.join(radiomics_folder, "cov2radiomics-Ts-FeatureSelection.csv")
# test_lesion_features_file_path = os.path.join(radiomics_folder, "cov2radiomics-Tr-FeatureSelection.csv")


# Select the model to evaluate
model_name = 'RandomForestClassifier'
                #'LogisticRegression'
                #'RandomForestClassifier'
                #'GradientBoostingClassifier'

## Read the dataset for training the model
ml_folder = os.path.join(testbed, experiment_name, "machine_learning/")
model_path = str(ml_folder+'/models/')

## Use oh_flat to encode labels as one-hot for RandomForestClassifier
oh_flat = True


##---------------------------------------------------------------------
## Load dataset
# X_test, y_test = load_features(test_lesion_features_file_path)

X_test, y_test, X_index = load_features_index(test_lesion_features_file_path)

print("---"*20)
print("X: {} || y: {} ".format(str(X_test.shape), str(y_test.shape)))
# print("X_index: {}".format(X_index.shape))




## ML evaluation
mlc = MLClassifier()
page_clf, test_score = mlc.model_evaluation(model_path, model_name, X_test, y_test, oh_flat)

y_predicted = page_clf.predict(X_test)
# print("X_index: {} || y_predicted: {}".format(type(X_index), type(y_predicted)))


## Concatenate the idcase and predicted
slices_pred_ = pd.DataFrame((X_index, y_predicted))
slices_predicted = pd.DataFrame(slices_pred_.T.values,
                                columns=['id_case', 'y_predicted'])
# print("DF: {}".format(slices_predicted))



## Groupby per slices_pred_
test_num_slices = slices_predicted.groupby(['id_case']).count()
test_num_slices = test_num_slices.reset_index()
# print("DF: {}".format(test_num_slices))

# value_counts = slices_predicted['id_case'].value_counts()
# print('test_num_slices', value_counts)


## Loop between the numbers
patient_predicted = []
patient_id = []
patient_percentage_slices_intubated = []
patient_percentage_slices_non_intubated = []


for row in test_num_slices.T.iteritems():
    print("---"*10)
    id_case = row[1][0]
    total_slices = row[1][1]

    predictions_per_case = slices_predicted.loc[lambda df: df['id_case'] == id_case]
    print('predictions_per_case', predictions_per_case)

    value_counts = predictions_per_case.value_counts()
    # print('test_num_slices', value_counts)
    # print('test_num_slices', value_counts.values)
    # print('Shape test_num_slices', value_counts.shape)

    non_intubated_slices = predictions_per_case.loc[(predictions_per_case['y_predicted'] == 0)]
    intubated_slices = predictions_per_case.loc[(predictions_per_case['y_predicted'] == 1)]


    try:
        num_non_intubated_slices = non_intubated_slices.value_counts()[0]
        num_intubated_slices = intubated_slices.value_counts()[0]

    except (IndexError):
        # num_non_intubated_slices = non_intubated_slices.value_counts()[0]
        # num_intubated_slices = 0
        try:
            num_non_intubated_slices = non_intubated_slices.value_counts()[0]
            num_intubated_slices = 0
        except (IndexError):
            print("non_intubated_slices", non_intubated_slices)
            print("intubated_slices", intubated_slices)



    print('non_intubated_slices', num_non_intubated_slices)
    print('intubated_slices', num_intubated_slices)

    non_intubated = num_non_intubated_slices
    intubated = num_intubated_slices

    ## Percentage of number of axial slices Predicted as Intubated
    percentage = int(intubated*100/(non_intubated + intubated))
    patient_percentage_slices_intubated.append(percentage)

    ## Percentage of number of axial slices Predicted as Non-Intubated
    non_intubated_percentage = int(non_intubated*100/(non_intubated + intubated))
    patient_percentage_slices_non_intubated.append(non_intubated_percentage)

    if int(percentage) >= int(60):
        patient_predicted.append(1)
        patient_id.append(id_case)

        print('intubated:  {} || {} || {}'.format(1, percentage, intubated))

    else:
        patient_predicted.append(0)
        patient_id.append(id_case)

        print('non_intubated: {} || {} || {}'.format(0, str(non_intubated_percentage), non_intubated))


##########################
## Save

print(patient_predicted)
print(patient_id)

## Concatenate the idcase and predicted
patients_pred_ = pd.DataFrame((patient_id, patient_predicted, patient_percentage_slices_intubated, patient_percentage_slices_non_intubated))
patients_pred = pd.DataFrame(patients_pred_.T.values,
                                columns=['id_case', 'GC_predicted', 'GC_percentage_slices_intubated', 'GC_percentage_slices_non_intubated'])
                                # columns=['id_case', 'MC_predicted', 'MC_percentage_slices_intubated', 'MC_percentage_slices_non_intubated'])


# patients_pred.to_csv('/home/jagh/Documents/01_UB/30_MedNeurIPS_2021/02_materials/output/multi-class_predictionTr.csv')
patients_pred.to_csv('/home/jagh/Documents/01_UB/30_MedNeurIPS_2021/02_materials/output/general-class_predictionTr.csv')

# print('patients_pred', patients_pred)
