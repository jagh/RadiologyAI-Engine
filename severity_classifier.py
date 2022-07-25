
import logging
import argparse

import sys, os
import glob
import pandas as pd
import numpy as np
import csv

from engine.utils import Utils
from engine.segmentations import LungSegmentations
from engine.featureextractor import RadiomicsExtractor
from engine.ml_classifier import MLClassifier
from engine.charts import Charts

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import plot_confusion_matrix

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics



def load_features(file_path):
    """
    Read features and labels per file.
    Step-1: From multiple files define features, labels and spliting the dataset
    """
    data = pd.read_csv(file_path, sep=',', header=0)
    ## Set features and labels, discard the two cases for a GGO 'CT-4'
    # X_data = data.values[:,3:].astype(np.float).astype("Int32")  #[:1107,2:]
    X_data = data.values[:,4:].astype(np.float)  #[:,3:]
    y_data = data.values[:,3]   #[:,2]
    y_data=y_data.astype('int')
    print("X_data: {} || y_data: {} ".format(str(X_data.shape), str(y_data.shape)))
    return X_data, y_data

def load_features_index(file_path):
    """Read features and labels per file"""
    data = pd.read_csv(file_path, sep=',', header=0)
    print(type(data))
    ## Set features and labels, discard the two cases for a GGO 'CT-4'
    # X_data = data.values[:,3:].astype(np.float).astype("Int32")  #[:1107,2:]
    X_data = data.values[:,4:].astype(np.float)  #[:,3:]
    y_data = data.values[:,3]   #[:,2]
    y_data = y_data.astype('int')
    X_index = data.values[:,1].astype('str')
    print("X_data: {} || y_data: {} ".format(str(X_data.shape), str(y_data.shape)))
    return X_data, y_data, X_index

def ml_grid_search(testbed, experiment_name, experiment_filename):
    ## Step-2: ML Training and Grid Search

    ## Accessing the data
    radiomics_folder = os.path.join(testbed, experiment_name, "radiomics_features/")
    lesion_features_file_path = os.path.join(radiomics_folder, experiment_filename)
    X_data, y_data = load_features(lesion_features_file_path)


    ## Create a ML folder and splitting the dataset
    ml_folder = os.path.join(testbed, experiment_name, "machine_learning/")
    Utils().mkdir(ml_folder)
    MLClassifier().splitting(X_data, y_data, ml_folder)


    ##-----------------------------------------------------------------------------
    ##-----------------------------------------------------------------------------
    ## Define location to write the model
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
    classifiers = [
                    LogisticRegression(**lr_params),
                    RandomForestClassifier(**rf_params),
                    # GradientBoostingClassifier(**gb_params),
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



################################################################################
## Evaluation Functions

from scipy.stats import entropy

def model_evaluation(testbed, experiment_name, experiment_filename, model_name='RandomForestClassifier'):

    ## General Lesion Full Features
    radiomics_folder = os.path.join(testbed, experiment_name, "radiomics_features/")
    test_lesion_features_file_path = os.path.join(radiomics_folder, experiment_filename)
                                        # "cov2radiomics-Ts-FeatureSelection.csv")

    ## Read the dataset for training the model
    ml_folder = os.path.join(testbed, experiment_name, "machine_learning/")
    model_path = str(ml_folder+'/models/')

    ## Use oh_flat to encode labels as one-hot for RandomForestClassifier
    oh_flat = True


    ##---------------------------------------------------------------------
    ## Load dataset
    X_test, y_test = load_features(test_lesion_features_file_path)

    # X_test = np.load(str(ml_folder+'/dataset/'+"/X_test_baseline.npy"), allow_pickle=True)
    # y_test = np.load(str(ml_folder+'/dataset/'+"/y_test_baseline.npy"), allow_pickle=True)

    print("---"*20)
    print("X_eval: {} || y_eval: {} ".format(str(X_test.shape), str(y_test.shape)))

    ## ML evaluation
    mlc = MLClassifier()
    page_clf, test_score = mlc.model_evaluation(model_path, model_name, X_test, y_test, oh_flat)


    ## Entropy measument
    predicted_probs = page_clf.predict_proba(X_test)  #important to use predict_proba
    print("+ Predicted_probs: ", predicted_probs.shape)
    print("+ X_test: ", X_test.shape)

    # predicted_entropy = entropy(predicted_probs, axis=1)
    # print("+ Predicted_entropy: ", predicted_entropy.shape)

    ## Plot the the confusion matrix by model selected
    # labels_name = ['3', '4', '5', '7', '6', '8', '9']
    labels_name = ['Non-Intubated', 'Intubated']
    plot_confusion_matrix(page_clf, X_test, y_test,
                                display_labels=labels_name,
                                cmap=plt.cm.Blues,
                                # normalize='true'
                                ) #, xticks_rotation=15)
    plt.title(str(model_name+" || F1-score: "+ str(test_score)))
    plt.show()


    ## ROC curves
    def plot_roc(model, X_test, y_test, model_name):
        # calculate the fpr and tpr for all thresholds of the classification
        probabilities = model.predict_proba(np.array(X_test))
        predictions = probabilities[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
        roc_auc = metrics.auc(fpr, tpr)

        plt.title(str(model_name + "") )
        plt.plot(fpr, tpr, 'g', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    plot_roc(page_clf, X_test, y_test, model_name)


from sklearn.metrics import f1_score

def model_entropy(testbed, experiment_name, experiment_filename, model_name='RandomForestClassifier'):

    ## General Lesion Full Features
    radiomics_folder = os.path.join(testbed, experiment_name, "radiomics_features/")
    test_lesion_features_file_path = os.path.join(radiomics_folder, experiment_filename)
                                        # "cov2radiomics-Ts-FeatureSelection.csv")

    ## Read the dataset for training the model
    ml_folder = os.path.join(testbed, experiment_name, "machine_learning/")
    model_path = str(ml_folder+'/models/')

    ## Use oh_flat to encode labels as one-hot for RandomForestClassifier
    oh_flat = True


    ##---------------------------------------------------------------------
    ## Load dataset
    # X_test, y_test = load_features(test_lesion_features_file_path)
    X_test, y_test, X_index = load_features_index(test_lesion_features_file_path)

    # X_test = np.load(str(ml_folder+'/dataset/'+"/X_test_baseline.npy"), allow_pickle=True)
    # y_test = np.load(str(ml_folder+'/dataset/'+"/y_test_baseline.npy"), allow_pickle=True)

    print("---"*20)
    print("X_eval: {} || y_eval: {} ".format(str(X_test.shape), str(y_test.shape)))

    ## ML evaluation
    mlc = MLClassifier()
    page_clf, test_score = mlc.model_evaluation(model_path, model_name, X_test, y_test, oh_flat)

    ## predictions has the format-> {non_intubated: 0.1522, intubated: 0.8477}
    y_predicted = page_clf.predict(X_test)

    print("y_test", type(y_test))
    print("y_predicted", type(y_predicted))
    y_score = f1_score(y_test, y_predicted, average='macro')
    print("+ score: ", y_score)

    ## Entropy measument
    predicted_probs = page_clf.predict_proba(X_test)

    print("X_test: ", X_test.shape)
    print("y_predicted: ", y_predicted.shape)
    print("predicted_probs: ", predicted_probs.shape)
    # print("predicted_probs: ", predicted_probs)


    ## Concatenate the idcase and predicted
    # slices_pred_ = pd.DataFrame((X_index, y_test, y_predicted, predicted_probs[:, 1]))
    slices_pred_ = pd.DataFrame((X_index, y_test, y_predicted, predicted_probs))
    slices_predicted = pd.DataFrame(slices_pred_.T.values,
                        columns=['id_case', 'y_test', 'y_predicted', 'predicted_probs'])


    ## Groupby per slices_pred_
    test_num_slices = slices_predicted.groupby(['id_case']).count()
    test_num_slices = test_num_slices.reset_index()


    ###############################################################################
    ## Loop between the numbers
    patient_predicted = []
    patient_id = []
    # cases_predicted = pd.DataFrame()
    cases_predicted = []
    for row in test_num_slices.T.iteritems():
        print("---"*10)
        id_case = row[1][0]
        total_slices = row[1][1]

        predictions_per_case = slices_predicted.loc[lambda df: df['id_case'] == id_case]
        # print('predictions_probabilities', predictions_per_case['predicted_probs'])


        ## Compute Entropy
        series_of_probabilities_per_case = predictions_per_case['predicted_probs'].values.tolist()
        # print("+ Series__probabilities", series_of_probabilities_per_case)
        # print("series_of_predictios_per_case", type(series_of_predictios_per_case))

        predicted_entropy = entropy(series_of_probabilities_per_case, axis=1)
        average_entropy = np.average(predicted_entropy)
        std_entropy = np.std(predicted_entropy)
        # print("+ Predicted_entropy: ", predicted_entropy)
        # print("+ Entropy AVG: {}, STD: {}".format(average_entropy, std_entropy))

        # case_pred_ = pd.DataFrame((X_index, y_test, y_predicted, average_entropy, std_entropy))
        # cases_predicted = cases_predicted.append(case_pred_)# assign it back

        ## Compute score
        y_test_series_per_case = predictions_per_case['y_test'].values.tolist()
        y_predicted_series_per_case = predictions_per_case['y_predicted'].values.tolist()

        y_score = f1_score(y_test_series_per_case, y_predicted_series_per_case, average='micro')
        print("+ score: ", y_score)

        cases_predicted.append((id_case, y_score, average_entropy, std_entropy))



    cases_predicted = pd.DataFrame(cases_predicted,
                            columns=['id_case', 'y_score', 'average_entropy', 'std_entropy'])

    print("cases_predicted", type(cases_predicted))
    print("cases_predicted", cases_predicted)

    metrics_folder = os.path.join(testbed, experiment_name, "metrics_folder/")
    Utils().mkdir(metrics_folder)
    cases_predicted.to_csv(os.path.join(metrics_folder, str(model_name +".csv")))


def feature_visualization(testbed, experiment_name, experiment_filename, model_name='RandomForestClassifier'):
    """"""

    ## Set file_path
    radiomics_folder = os.path.join(testbed, experiment_name, "radiomics_features/")
    lesion_features_file_path = os.path.join(radiomics_folder, experiment_filename)

    ## Get features, labels and patient index
    data = pd.read_csv(lesion_features_file_path, sep=',', header=0)

    ## Set features to visualize
    X_data = data.iloc[:,3:]

    ## Set Folder
    visualizaion_folder = os.path.join(testbed, experiment_name, "visualization_features/")
    Utils().mkdir(visualizaion_folder)


    ## Plot a hierarchical cluster
    Charts().plot_heatmap(X_data, visualizaion_folder, model_name)


def plot_radiomic_features(testbed, experiment_name, experiment_filename, model_name='RandomForestClassifier'):
    """
    PyRadiomic Features:
        0 -> A. Intensity Histogram       First-order statistics
        1 -> B. Shape Features (2D)      Two-dimensional size and shape of the ROI
        2 -> C. GLCM Features:           Gray Level Co-occurrence Matrix
        3 -> D. GLSZM Features:          Gray Level Size Zone Matrix
        4 -> E. NGTDM Features:          Neighbouring Gray Tone Difference Matrix
        5 -> F. GLDM Features:           Gray Level Dependence Matrix
        6 -> G. GLRLM Features:          Gray Level Run Length Matrix
    """


    ## Set file_path
    radiomics_folder = os.path.join(testbed, experiment_name, "radiomics_features/")
    lesion_features_file_path = os.path.join(radiomics_folder, experiment_filename)

    ## Get features, labels and patient index
    data = pd.read_csv(lesion_features_file_path, sep=',', header=0)

    for i in range(7):
        if i == 0:
            ## Set features to visualize
            X_data = data.loc[:,'AA_10Percentile':'AA_Variance']
            fig_name = 'intensity_histogram'
        elif i == 1:
            ## Set features to visualize
            X_data = data.loc[:,'BB_Elongation':'BB_Sphericity']
            fig_name = 'shape_features'
        elif i == 2:
            ## Set features to visualize
            X_data = data.loc[:,'CC_Autocorrelation':'CC_SumSquares']
            fig_name = 'glcm_features'
        elif i == 3:
            ## Set features to visualize
            X_data = data.loc[:,'DD_GrayLevelNonUniformity':'DD_ShortRunLowGrayLevelEmphasis']
            fig_name = 'glszm_features'
        elif i == 4:
            ## Set features to visualize
            X_data = data.loc[:,'EE_Busyness':'EE_Strength']
            fig_name = 'gldm_features'
        elif i == 5:
            ## Set features to visualize
            X_data = data.loc[:,'FF_DependenceEntropy':'FF_SmallDependenceLowGrayLevelEmphasis']
            fig_name = 'gldm_features'
        elif i == 6:
            ## Set features to visualize
            X_data = data.loc[:,'GG_GrayLevelNonUniformity':'GG_ZoneVariance']
            fig_name = 'glrlm_features'
        else:
            print(i)


        ## Set Folder
        visualizaion_folder = os.path.join(testbed, experiment_name, "visualization_features/")
        Utils().mkdir(visualizaion_folder)


        ## Plot a hierarchical cluster
        Charts().plot_heatmap(X_data, visualizaion_folder, fig_name)


def model_evaluation_patients(testbed, experiment_name, experiment_filename, model_name='RandomForestClassifier', output_file='PatientLevelTr'):
    """ Evaluation at the patient level """

    ## General Lesion Full Features
    radiomics_folder = os.path.join(testbed, experiment_name, "radiomics_features/")
    features_set_file_path = os.path.join(radiomics_folder, experiment_filename)

    ## Read the dataset for training the model
    ml_folder = os.path.join(testbed, experiment_name, "machine_learning/")
    model_path = str(ml_folder+'/models/')

    ## Use oh_flat to encode labels as one-hot for RandomForestClassifier
    oh_flat = True


    ##---------------------------------------------------------------------
    ## Load dataset
    X_test, y_test, X_index = load_features_index(features_set_file_path)
    # print("X_eval: {} || y_eval: {} ".format(str(X_test.shape), str(y_test.shape)))

    ## ML evaluation
    mlc = MLClassifier()
    page_clf, test_score = mlc.model_evaluation(model_path, model_name, X_test, y_test, oh_flat)

    ## predictions has the format-> {non_intubated: 0.1522, intubated: 0.8477}
    y_predicted = page_clf.predict(X_test)

    # print("y_test", type(y_test))
    # print("y_predicted", type(y_predicted))
    ## y_score = f1_score(y_test, y_predicted, average='macro')
    y_score = f1_score(y_test, y_predicted, average='micro')
    print("+ Micro F1-score: ", y_score)

    ## Entropy measument
    predicted_probs = page_clf.predict_proba(X_test)
    # print("X_test: ", X_test.shape)
    # print("y_predicted: ", y_predicted.shape)
    # print("predicted_probs: ", predicted_probs.shape)


    ## Concatenate the id_case and predicted
    # slices_pred_ = pd.DataFrame((X_index, y_test, y_predicted, predicted_probs[:, 1]))
    slices_pred_ = pd.DataFrame((X_index, y_test, y_predicted, predicted_probs))
    slices_predicted = pd.DataFrame(slices_pred_.T.values,
                        columns=['id_case', 'y_test', 'y_predicted', 'predicted_probs'])
    # print("+ Slices_predicted: ", slices_predicted)


    ## Groupby per slices_pred_
    test_num_slices = slices_predicted.groupby(['id_case']).count()
    test_num_slices = test_num_slices.reset_index()
    # print("+ test_num_slices: ", test_num_slices)



    ###############################################################################
    ## Loop between the numbers
    patient_predicted = []
    patient_id = []
    # cases_predicted = pd.DataFrame()
    cases_predicted = []
    for row in test_num_slices.T.iteritems():
        # print("---"*10)
        id_case = row[1][0]
        total_slices = row[1][1]
        # print("+ id_case: ", id_case)
        # print("+ total_slices: ", total_slices)

        ## Step-1: Collecting slice-wise predictions by case
        predictions_per_case = slices_predicted.loc[lambda df: df['id_case'] == id_case]
        # print('+ Predictions per case: ', predictions_per_case['y_predicted'].shape)

        ## Step-2: Grouping and counting the predictions by each who class
        predictions_per_case_groupby = predictions_per_case.groupby(['y_predicted']).count()
        # print('+ Groupby Predictions per slice: ', predictions_per_case_groupby)

        ## Step-3: Parsing the who-class-wise count
        predictions_per_case_groupby.reset_index(inplace=True)
        predictions_per_case_values = predictions_per_case_groupby.values
        # print("+ predictions_per_case_values: ", type(predictions_per_case_values))
        # print("+ predictions_per_case_values: ", predictions_per_case_values[:, 1])

        ## Step-4: Get max votes and get the index of the who class
        max_votes = predictions_per_case_values[:, 1].max(axis=0)
        index_who_score = np.where(predictions_per_case_values[:, 1] == max_votes)
        # print("+ index_who_score: ", index_who_score[0][0])

        ##########################################################
        ## Step-5: Get the majority voting of who score by index
        list_of_predictions = predictions_per_case_values[:, 0].tolist()
        majority_voting_prediction = list_of_predictions[index_who_score[0][0]]
        # print("+ list_of_predictions: ", who_prediction_patient)


        ##########################################################
        ## Step-1: Get the mean voting of who score
        mean_voting_prediction = np.average(predictions_per_case['y_predicted'].values)
        # print("+ list_of_predictions: ", predictions_per_case['y_predicted'].values)
        # print("+ mean_voting_prediction: ", mean_voting_prediction)


        ##########################################################
        ## Step-1: Get the GT per case and get the list of predictions per slices
        GT_per_case = predictions_per_case['y_test'].values
        # print("+ Predictions_per_case: ", GT_per_case[0])

        ## Step-2: Get the additional information
        list_of_predictions_per_slices = predictions_per_case['y_predicted'].values
        # print("+ List_of_predictios_per_slices: ", list_of_predictios_per_slices)



        cases_predicted.append((id_case, GT_per_case[0], majority_voting_prediction, mean_voting_prediction ,total_slices, list_of_predictions_per_slices))




    cases_predicted = pd.DataFrame(cases_predicted,
                            columns=['id_case', 'GT_per_case', 'majority_voting_prediction', 'mean_voting_prediction', 'total_slices', 'list_of_predictions_per_slices'])


    # print("+ DataFrame: ", cases_predicted)
    print(cases_predicted)

    metrics_folder = os.path.join(testbed, experiment_name, "metrics_folder/")
    Utils().mkdir(metrics_folder)
    cases_predicted.to_csv(os.path.join(metrics_folder, str(model_name+"-"+output_file+"-sample.csv")))


    ##########################################
    ## Compute confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    ## Step-1: Set labels
    labels_name = ['3', '4', '5', '7', '6', '8', '9']
    # labels_name = ['Non-Intubated', 'Intubated']
    ##########################################
    ## Step-2.1: Compute the F1-score per patient
    majority_voting_F1score = np.round_(f1_score(cases_predicted['GT_per_case'].values,
                                    cases_predicted['majority_voting_prediction'].values,
                                    average='micro'), decimals=3)
    print("+ majority_voting_F1score: ", majority_voting_F1score)

    ## Step-2.2: Compute the confusion matrix by majority_voting_prediction
    majority_voting_CM = confusion_matrix(y_true=cases_predicted['GT_per_case'].values,
                                            y_pred=cases_predicted['majority_voting_prediction'].values,
                                            # labels=labels_name,
                                            )
    # print("+ CM by majority_voting_prediction: ", majority_voting_CM)

    ## Step-2.3: Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=majority_voting_CM,
                                    display_labels=labels_name,
                                    )
    disp.plot()
    plt.title(str(model_name+" || F1-score: "+ str(majority_voting_F1score)))
    plt.show()



    # ##########################################
    # ## Step-2: Compute the confusion matrix by mean_voting_prediction
    # labels_name_2 = ['3', '4', '5', '6', '7', '8', '9']
    # # labels_name_2 = ['3', '4', '5', '6', '8', '9']
    #
    # ## Step-1: Compute the F1-score per patient
    # mean_voting_F1score = np.round_(f1_score(cases_predicted['GT_per_case'].values,
    #                                 cases_predicted['mean_voting_prediction'].values,
    #                                 average='micro'), decimals=3)
    # print("+ majority_voting_F1score: ", majority_voting_F1score)
    #
    # mean_voting_CM = confusion_matrix(y_true=cases_predicted['GT_per_case'].values,
    #                                         y_pred=cases_predicted['mean_voting_prediction'].values,
    #                                         # labels=labels_name,
    #                                         )
    # # print("+ CM by mean_voting_prediction: ", mean_voting_CM)
    #
    # ## Step-2.1: Display the confusion matrix
    # disp = ConfusionMatrixDisplay(confusion_matrix=mean_voting_CM,
    #                                 display_labels=labels_name_2,
    #                                 )
    # disp.plot()
    # plt.title(str(model_name+" || F1-score: "+ str(mean_voting_F1score)))
    # plt.show()
    print("+ Process completed!")


################################################################################
################################################################################
## XGBoost Classifier Functions
import pickle
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import f1_score

def stratifiedShuffleSplit(testbed, experiment_name, experiment_filename):
    ## Step-2: ML Training and Grid Search

    ## Accessing the data
    radiomics_folder = os.path.join(testbed, experiment_name, "radiomics_features/")
    features_file_path = os.path.join(radiomics_folder, experiment_filename)


    ## Load_features_index
    data = pd.read_csv(features_file_path, sep=',', header=0)
    ## Set features and labels, discard the two cases for a GGO 'CT-4'
    # X_data = data.values[:,3:].astype(np.float).astype("Int32")  #[:1107,2:]
    X_data = data.values[:,1:]  #.astype(np.float)  #[:1107,2:]
    y_data = data.values[:,0]   #[:1107,1]
    # y_data=y_data.astype('int')

    ## Create a ML folder and splitting the dataset
    eval_split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
    for train_index, test_index in eval_split.split(X_data, y_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        # print("train_index: {} || test_index: {} ".format(str(train_index.shape), str(test_index.shape) ))

    print("X_train: {} || y_train: {} ".format(str(X_train.shape), str(y_train.shape)))
    print("X_test: {} || y_test: {} ".format(str(X_test.shape), str(y_test.shape) ))


    ####################
    dataframeTr = pd.DataFrame(y_train, X_train)
    dataframeTr_file_path = os.path.join(radiomics_folder, "3DgenerealFF_dataframeTr")
    dataframeTr.to_csv(dataframeTr_file_path, sep=',', index=True)
    # print("dataframeTr: ", dataframeTr)


    dataframeTs = pd.DataFrame(y_test, X_test)
    dataframeTs_file_path = os.path.join(radiomics_folder, "3DgenerealFF_dataframeTs")
    dataframeTs.to_csv(dataframeTs_file_path, sep=',', index=True)
    print("dataframeTs: ", dataframeTs)

def xgboostTraining(testbed, experiment_name, experiment_filename):
    ## Step-2: ML Training and Grid Search

    ## General Lesion Full Features
    radiomics_folder = os.path.join(testbed, experiment_name, "radiomics_features/")
    train_lesion_features_file_path = os.path.join(radiomics_folder, experiment_filename)
                                        # "cov2radiomics-Ts-FeatureSelection.csv")

    ## Read the dataset for training the model
    ml_folder = os.path.join(testbed, experiment_name, "machine_learning/")
    model_path = str(ml_folder+'/models/')

    ## Use oh_flat to encode labels as one-hot for RandomForestClassifier
    oh_flat = True


    ##---------------------------------------------------------------------
    ## Load dataset
    X, y = load_features(train_lesion_features_file_path)

    # rng = np.random.RandomState(31337)
    # kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    # for train_index, test_index in kf.split(X):

    # test_sizes = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10]
    # test_sizes = [0.15, 0.10]
    this_cv = StratifiedShuffleSplit(n_splits=4, test_size=0.10, random_state=0)
    for train_index, test_index in this_cv.split(X, y):
        # xgb_model = xgb.XGBClassifier(n_jobs=1).fit(X[train_index], y[train_index])
        xgb_model = lgb.LGBMClassifier(n_jobs=1).fit(X[train_index], y[train_index])

        ## Writing the developing model
        model_file = open(str(model_path+"/sgb_model.pkl"), "wb")
        pickle.dump(xgb_model, model_file)
        model_file.close()


        ## Training performance
        predictions = xgb_model.predict(X[test_index])
        actuals = y[test_index]

        train_score = f1_score(actuals, predictions, average='macro')
        print("++ Train F1-Score: {}".format(train_score))
        print(confusion_matrix(actuals, predictions))


        ## Plot the the confusion matrix by model selected
        # labels_name = ['Non-Intubated', 'Intubated']
        # labels_name = ['3', '4', '5', '6', '8', '9']
        # labels_name = ['3', '4', '5', '6', '8', '9']
        # plot_confusion_matrix(xgb_model, X, y,
        #                             display_labels=labels_name,
        #                             cmap=plt.cm.Blues,
        #                             # normalize='true'
        #                             ) #, xticks_rotation=15)
        # plt.title(str(" XGBoost || F1-score: "+ str(train_score)))
        # plt.show()



        ##---------------------------------------------------------------------
        ## Load dataset
        test_lesion_features_file_path = os.path.join(radiomics_folder, "cov2radiomics-Ts-FeatureSelection-WHO.csv")
        X_test, y_test = load_features(test_lesion_features_file_path)

        ## Training performance
        predictions = xgb_model.predict(X_test)
        actuals = y_test

        test_score = f1_score(actuals, predictions, average='macro')
        print("++ Test F1-Score: {}".format(test_score))
        print(confusion_matrix(actuals, predictions))


        ## Plot the the confusion matrix by model selected
        # labels_name = ['Non-Intubated', 'Intubated']
        # labels_name = ['3', '4', '5', '6', '7', '8', '9']
        labels_name = ['Mild', 'Moderate', 'Severe']
        plot_confusion_matrix(xgb_model, X_test, y_test,
                                    display_labels=labels_name,
                                    cmap=plt.cm.Blues,
                                    # normalize='true'
                                    ) #, xticks_rotation=15)
        plt.title(str(" XGBoost || F1-score: "+ str(test_score)))
        plt.show()

def xgboostTraining_evaluationPatients(testbed, experiment_name, experiment_filename):
    ## Step-2: ML Training and Grid Search

    model_name='XGBoostClassifier'

    ## General Lesion Full Features
    radiomics_folder = os.path.join(testbed, experiment_name, "radiomics_features/")
    train_lesion_features_file_path = os.path.join(radiomics_folder, experiment_filename)
                                        # "cov2radiomics-Ts-FeatureSelection.csv")

    ## Read the dataset for training the model
    ml_folder = os.path.join(testbed, experiment_name, "machine_learning/")
    model_path = str(ml_folder+'/models/')

    ## Use oh_flat to encode labels as one-hot for RandomForestClassifier
    oh_flat = True


    ##---------------------------------------------------------------------
    ## Load dataset
    X, y = load_features(train_lesion_features_file_path)

    # rng = np.random.RandomState(31337)
    # kf = KFold(n_splits=2, shuffle=True, random_state=rng)
    # for train_index, test_index in kf.split(X):

    # test_sizes = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10]
    # test_sizes = [0.15, 0.10]
    splits_count = 1
    this_cv = StratifiedShuffleSplit(n_splits=6, test_size=0.15, random_state=0)
    for train_index, test_index in this_cv.split(X, y):
        # xgb_model = xgb.XGBClassifier(n_jobs=8).fit(X[train_index], y[train_index])
        xgb_model = xgb.XGBClassifier(n_jobs=-1,
        # xgb_model = lgb.LGBMClassifier(n_jobs=-1,
                                        base_score=0.2,
                                        booster='gbtree',
                                        gamma= 0,
                                        learning_rate= 0.1,
                                        n_estimators=500,
                                        reg_alpha=0,
                                        reg_lambda=1,
                                        num_classes=3)
        xgb_model = xgb_model.fit(X[train_index], y[train_index])

        ## Writing the developing model
        model_file = open(str(model_path+"/sgb_model.pkl"), "wb")
        pickle.dump(xgb_model, model_file)
        model_file.close()


        ## Training performance
        predictions = xgb_model.predict(X[test_index])
        actuals = y[test_index]

        train_score = f1_score(actuals, predictions, average='macro')
        print("++ Train F1-Score: {}".format(train_score))
        # print(confusion_matrix(actuals, predictions))


        # ## Plot the the confusion matrix by model selected
        # # labels_name = ['Non-Intubated', 'Intubated']
        # labels_name = ['3', '4', '5', '6', '8', '9']
        # # labels_name = ['4', '5', '6', '8', '9']
        # plot_confusion_matrix(xgb_model, X, y,
        #                             display_labels=labels_name,
        #                             cmap=plt.cm.Blues,
        #                             # normalize='true'
        #                             ) #, xticks_rotation=15)
        # plt.title(str(" XGBoost || F1-score: "+ str(train_score)))
        # plt.show()


        # ##---------------------------------------------------------------------
        # ## Load dataset Validation per slice
        # test_lesion_features_file_path = os.path.join(radiomics_folder, "cov2radiomics-Ts-FeatureSelection-WHO.csv")
        # X_test, y_test = load_features(test_lesion_features_file_path)
        #
        # ## Training performance
        # predictions = xgb_model.predict(X_test)
        # actuals = y_test
        #
        # test_score = f1_score(actuals, predictions, average='macro')
        # print("++ Test F1-Score: {}".format(test_score))
        # print(confusion_matrix(actuals, predictions))
        #
        #
        # ## Plot the the confusion matrix by model selected
        # # labels_name = ['Non-Intubated', 'Intubated']
        # labels_name = ['3', '4', '5', '6', '8', '9']
        # plot_confusion_matrix(xgb_model, X_test, y_test,
        #                             display_labels=labels_name,
        #                             cmap=plt.cm.Blues,
        #                             # normalize='true'
        #                             ) #, xticks_rotation=15)
        # plt.title(str(" XGBoost || F1-score: "+ str(test_score)))
        # plt.show()



        ##---------------------------------------------------------------------
        ## Patiente Level Evaluation -
        ## Load the test set
        test_lesion_features_file_path = os.path.join(radiomics_folder, "cov2radiomics-Ts-FeatureSelection-WHO.csv")
        X_test, y_test, X_index = load_features_index(test_lesion_features_file_path)
        # print("X_eval: {} || y_eval: {} ".format(str(X_test.shape), str(y_test.shape)))

        ## ML evaluation
        # mlc = MLClassifier()
        # page_clf, test_score = mlc.model_evaluation(model_path, model_name, X_test, y_test, oh_flat)


        ## Get predictions from XGB Model
        y_predicted = xgb_model.predict(X_test)

        ## y_score = f1_score(y_test, y_predicted, average='macro')
        y_score = f1_score(y_test, y_predicted, average='micro')
        print("+ Micro F1-score: ", y_score)

        ## Entropy measument
        # predicted_probs = page_clf.predict_proba(X_test)
        # print("X_test: ", X_test.shape)
        # print("y_predicted: ", y_predicted.shape)
        # print("predicted_probs: ", predicted_probs.shape)


        ## Concatenate the id_case and predicted
        slices_pred_ = pd.DataFrame((X_index, y_test, y_predicted))
        slices_predicted = pd.DataFrame(slices_pred_.T.values,
                            columns=['id_case', 'y_test', 'y_predicted'])


        ## Groupby per slices_pred_
        test_num_slices = slices_predicted.groupby(['id_case']).count()
        test_num_slices = test_num_slices.reset_index()
        print("+ test_num_slices: ", test_num_slices)


        ###############################################################################
        ## Loop between the numbers
        patient_predicted = []
        patient_id = []
        # cases_predicted = pd.DataFrame()
        cases_predicted = []
        for row in test_num_slices.T.iteritems():
            # print("---"*10)
            id_case = row[1][0]
            total_slices = row[1][1]
            # print("+ id_case: ", id_case)
            # print("+ total_slices: ", total_slices)

            ## Step-1: Collecting slice-wise predictions by case
            predictions_per_case = slices_predicted.loc[lambda df: df['id_case'] == id_case]
            # print('+ Predictions per case: ', predictions_per_case['y_predicted'].shape)

            ## Step-2: Grouping and counting the predictions by each who class
            predictions_per_case_groupby = predictions_per_case.groupby(['y_predicted']).count()
            # print('+ Groupby Predictions per slice: ', predictions_per_case_groupby)

            ## Step-3: Parsing the who-class-wise count
            predictions_per_case_groupby.reset_index(inplace=True)
            predictions_per_case_values = predictions_per_case_groupby.values
            # print("+ predictions_per_case_values: ", type(predictions_per_case_values))
            # print("+ predictions_per_case_values: ", predictions_per_case_values[:, 1])

            ## Step-4: Get max votes and get the index of the who class
            max_votes = predictions_per_case_values[:, 1].max(axis=0)
            index_who_score = np.where(predictions_per_case_values[:, 1] == max_votes)
            # print("+ index_who_score: ", index_who_score[0][0])

            ##########################################################
            ## Step-5: Get the majority voting of who score by index
            list_of_predictions = predictions_per_case_values[:, 0].tolist()
            majority_voting_prediction = list_of_predictions[index_who_score[0][0]]
            # print("+ list_of_predictions: ", who_prediction_patient)


            ##########################################################
            ## Step-1: Get the mean voting of who score
            mean_voting_prediction = np.average(predictions_per_case['y_predicted'].values)
            # print("+ list_of_predictions: ", predictions_per_case['y_predicted'].values)
            # print("+ mean_voting_prediction: ", mean_voting_prediction)


            ##########################################################
            ## Step-1: Get the GT per case and get the list of predictions per slices
            GT_per_case = predictions_per_case['y_test'].values
            # print("+ Predictions_per_case: ", GT_per_case[0])

            ## Step-2: Get the additional information
            list_of_predictions_per_slices = predictions_per_case['y_predicted'].values
            # print("+ List_of_predictios_per_slices: ", list_of_predictios_per_slices)


            cases_predicted.append((id_case, GT_per_case[0], majority_voting_prediction, mean_voting_prediction ,total_slices, list_of_predictions_per_slices))




        cases_predicted = pd.DataFrame(cases_predicted,
                                columns=['id_case', 'GT_per_case', 'majority_voting_prediction', 'mean_voting_prediction', 'total_slices', 'list_of_predictions_per_slices'])


        # print("+ DataFrame: ", cases_predicted)
        print(cases_predicted)

        splits_count = splits_count + 1
        print("+ splits_count: ", splits_count)


        ##########################################
        ## Compute confusion matrix
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        ## Step-1: Set labels
        # labels_name = ['3', '4', '5', '6', '7', '8', '9']
        labels_name = ['Mild', 'Moderate', 'Severe']
        # labels_name = ['0', '1']
        ##########################################
        ## Step-2.1: Compute the F1-score per patient
        majority_voting_F1score = np.round_(f1_score(cases_predicted['GT_per_case'].values,
                                        cases_predicted['majority_voting_prediction'].values,
                                        # average='micro'), decimals=3)
                                        average='micro'), decimals=2)
        print("+ majority_voting_F1score: ", majority_voting_F1score)


        ###########################################
        ## Write
        metrics_folder = os.path.join(testbed, experiment_name, "metrics_folder/")
        Utils().mkdir(metrics_folder)
        cases_predicted.to_csv(os.path.join(metrics_folder, str(model_name + str(splits_count) + "-F1_" + str(majority_voting_F1score) +"-PatientLevelTs.csv")))


        ## Step-2.2: Compute the confusion matrix by majority_voting_prediction
        majority_voting_CM = confusion_matrix(y_true=cases_predicted['GT_per_case'].values,
                                                y_pred=cases_predicted['majority_voting_prediction'].values,
                                                # labels=labels_name,
                                                )
        # print("+ CM by majority_voting_prediction: ", majority_voting_CM)

        ## Step-2.3: Display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=majority_voting_CM,
                                        display_labels=labels_name,
                                        )
        disp.plot()
        plt.title(str(model_name+" || F1-score: "+ str(majority_voting_F1score)))
        plt.show()


        # ##########################################
        # ## Step-2: Compute the confusion matrix by mean_voting_prediction
        # # labels_name_2 = ['3', '4', '5', '6', '7', '8', '9']
        # labels_name_2 = ['3', '4', '5', '6', '7', '8', '9']
        #
        # ## Step-1: Compute the F1-score per patient
        # mean_voting_F1score = np.round_(f1_score(cases_predicted['GT_per_case'].values,
        #                                 cases_predicted['mean_voting_prediction'].values,
        #                                 average='micro'), decimals=3)
        # print("+ majority_voting_F1score: ", majority_voting_F1score)
        #
        # mean_voting_CM = confusion_matrix(y_true=cases_predicted['GT_per_case'].values,
        #                                         y_pred=cases_predicted['mean_voting_prediction'].values,
        #                                         # labels=labels_name,
        #                                         )
        # # print("+ CM by mean_voting_prediction: ", mean_voting_CM)
        #
        # ## Step-2.1: Display the confusion matrix
        # disp = ConfusionMatrixDisplay(confusion_matrix=mean_voting_CM,
        #                                 display_labels=labels_name_2,
        #                                 )
        # disp.plot()
        # plt.title(str(model_name+" || F1-score: "+ str(mean_voting_F1score)))
        # plt.show()
        print("+ Not processing confusion matrix by mean_voting_prediction!")



###################################################################
def model_evaluation_slicesTr(testbed, experiment_name, test_set_filename, model_name='RandomForestClassifier'):
    """ Evaluation at the patient level """

    ## General Lesion Full Features
    radiomics_folder = os.path.join(testbed, experiment_name, "radiomics_features/")
    test_lesion_features_file_path = os.path.join(radiomics_folder, test_set_filename)

    ## Read the dataset for training the model
    ml_folder = os.path.join(testbed, experiment_name, "machine_learning/")
    model_path = str(ml_folder+'/models/')

    ## Use oh_flat to encode labels as one-hot for RandomForestClassifier
    oh_flat = True


    ##---------------------------------------------------------------------
    ## Load dataset
    X_test, y_test, X_index = load_features_index(test_lesion_features_file_path)
    # print("X_eval: {} || y_eval: {} ".format(str(X_test.shape), str(y_test.shape)))

    ## ML evaluation
    mlc = MLClassifier()
    page_clf, test_score = mlc.model_evaluation(model_path, model_name, X_test, y_test, oh_flat)

    ## predictions has the format-> {non_intubated: 0.1522, intubated: 0.8477}
    y_predicted = page_clf.predict(X_test)

    # print("y_test", type(y_test))
    # print("y_predicted", type(y_predicted))
    ## y_score = f1_score(y_test, y_predicted, average='macro')
    y_score = f1_score(y_test, y_predicted, average='micro')
    print("+ Micro F1-score: ", y_score)

    ## Entropy measument
    predicted_probs = page_clf.predict_proba(X_test)
    # print("X_test: ", X_test.shape)
    # print("y_predicted: ", y_predicted.shape)
    # print("predicted_probs: ", predicted_probs.shape)


    ## Concatenate the id_case and predicted
    # slices_pred_ = pd.DataFrame((X_index, y_test, y_predicted, predicted_probs[:, 1]))
    slices_pred_ = pd.DataFrame((X_index, y_test, y_predicted, X_test))
    slices_predicted = pd.DataFrame(slices_pred_.T.values,
                        columns=['id_case', 'y_test', 'y_predicted', 'X_test'])
    print("+ Slices_predicted: ", slices_predicted)

    metrics_folder = os.path.join(testbed, experiment_name, "metrics_folder/")
    Utils().mkdir(metrics_folder)
    slices_predicted.to_csv(os.path.join(metrics_folder, str(model_name +"-slices_predicted-Tr.csv")))



def run(args):
    ###################################
    # testbed = "testbed-WHO-20220325/"
    # testbed = "testbed-WHO-20220421/"
    # testbed = "testbed-WHO-20220427"

    ###################################
    # testbed = "testbed-WHO-20220427-NotNormalized"
    # testbed = "testbed-WHO-20220502-ClassBalanced"
    # testbed = "testbed-WHO-20220502"

    ###################################
    # testbed = "testbed-WHO-20220502-Intubation"

    ###################################
    # testbed = "testbed-WHO-20220503-CaseExchange"
    # testbed = "testbed-WHO-20220503-PosterResults-WHO"
    # testbed = "testbed-WHO-20220503-PosterResults-WHO-3Classes"

    ###################################
    # testbed = "testbed-WHO-01_RF-Exp1_Non-normalized"
    # testbed = "testbed-WHO-02_RF-Exp2_Z-score_Normalization"
    testbed = "testbed-WHO-02_RF-Exp3_step_Normalization"

    ## Features selected
    experiment_name = "02_MULTICLASS-PLUS" #"02_MULTICLASS" #"01_GENERALCLASS" #"04_GGO" #"03_CON" #"02_MULTICLASS-PLUS"
    train_set_filename = "cov2radiomics-Tr-FeatureSelection-WHO.csv"
    test_set_filename = "cov2radiomics-Ts-FeatureSelection-WHO.csv"

    # Select the model to evaluate
    model_name = 'RandomForestClassifier'
                    #'LogisticRegression'
                    #'RandomForestClassifier'
                    #'GradientBoostingClassifier'

    ## 00 - StratifiedShuffleSplit
    # stratifiedShuffleSplit(testbed, experiment_name, experiment_filename)

    ## 01 - Train
    # ml_grid_search(testbed, experiment_name, train_set_filename)

    # 02 - Test at the slice level
    # model_evaluation(testbed, experiment_name, test_set_filename, model_name)

    ## 03 - Test at the patient level
    # model_evaluation_patients(testbed, experiment_name, test_set_filename,
    #                             model_name, output_file='PatientLevelTs')

    ## 03.1 - Train
    # model_evaluation_slicesTr(testbed, experiment_name, train_set_filename, model_name)

    # model_evaluation_patients(testbed, experiment_name, train_set_filename, model_name, output_file='PatientLevelTr')

    ## 04 -- Alternative
    # xgboostTraining(testbed, experiment_name, train_set_filename)

    ## 04.1
    xgboostTraining_evaluationPatients(testbed, experiment_name, train_set_filename)

    ## 05 - Reliability metrics
    # model_entropy(testbed, experiment_name, experiment_filename, model_name)

    ## 06 - Feature Vsualization
    # feature_visualization(testbed, experiment_name, experiment_filename, model_name)
    # plot_radiomic_features(testbed, experiment_name, experiment_filename, model_name)

    ## 07 - Lasso ensemble for explainer
    # shap_explainer(testbed, experiment_name, experiment_filename, model_name)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='/data/01_UB/Multiomics-Data/Clinical_Imaging/02_Step-3_122-CasesSegmented/02_Nifti-Seg-6-Classes/')
    parser.add_argument('-s', '--sandbox', default='/data/01_UB/Multiomics-Data/Clinical_Imaging/02_Step-3_122-CasesSegmented/')

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    main()
