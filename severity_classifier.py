
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
    X_data = data.values[:,3:].astype(np.float)  #[:1107,2:]
    y_data = data.values[:,2]   #[:1107,1]
    y_data=y_data.astype('int')
    print("X_data: {} || y_data: {} ".format(str(X_data.shape), str(y_data.shape)))
    return X_data, y_data

def load_features_index(file_path):
    """Read features and labels per file"""
    data = pd.read_csv(file_path, sep=',', header=0)
    print(type(data))
    ## Set features and labels, discard the two cases for a GGO 'CT-4'
    # X_data = data.values[:,3:].astype(np.float).astype("Int32")  #[:1107,2:]
    X_data = data.values[:,3:].astype(np.float)  #[:1107,2:]
    y_data = data.values[:,2]   #[:1107,1]
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

    predicted_entropy = entropy(predicted_probs, axis=1)
    print("+ Predicted_entropy: ", predicted_entropy.shape)


    ## Plot the the confusion matrix by model selected
    labels_name = ['3', '4', '5', '6', '8', '9']
    # labels_name = ['Non-Intubated', 'Intubated']
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


import xgboost
import shap

import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import shap
import time

import chart_studio.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objects as go


from sklearn.ensemble import RandomForestRegressor
import xgboost
from bs4 import BeautifulSoup


def shap_explainer(testbed, experiment_name, experiment_filename, model_name='RandomForestClassifier'):
    """
        + Iris Classification with sckit-learn
    """


    ###########################################################################
    ## Training Set
    ## General Lesion Full Features
    radiomics_folder = os.path.join(testbed, experiment_name, "radiomics_features/")
    # train_lesion_features_file_path = os.path.join(radiomics_folder, "cov2radiomics-Tr-FeatureSelection.csv")
    train_lesion_features_file_path = os.path.join(radiomics_folder, "general2class-Tr-FeatureSelection-2.csv")


    ## Load training set
    data = pd.read_csv(train_lesion_features_file_path, sep=',', header=0)
    X_train = data.iloc[:,3:]  #[:1107,2:]
    y_train = data.iloc[:,2]   #[:1107,1]
    X_index_Tr = data.iloc[:,1].astype('str')

    print("---"*20)
    # print("X_data: ", X_data)
    print("X_data: {} || y_data: {} ".format(str(X_train.shape), str(y_train.shape)))



    ###########################################################################
    ## Test Set
    ## General Lesion Full Features
    radiomics_folder = os.path.join(testbed, experiment_name, "radiomics_features/")
    test_lesion_features_file_path = os.path.join(radiomics_folder, experiment_filename)

    ## Load training set
    data = pd.read_csv(test_lesion_features_file_path, sep=',', header=0)
    X_test = data.iloc[:,3:]  #[:1107,2:]
    y_test = data.iloc[:,2]   #[:1107,1]
    X_index_Ts = data.iloc[:,1].astype('str')

    print("---"*20)
    print("X_data: ", X_test)
    print("X_data: {} || y_data: {} ".format(str(X_test.shape), str(y_test.shape)))


    ##########################################################################
    ## ML evaluation

    ## Read the dataset for training the model
    ml_folder = os.path.join(testbed, experiment_name, "machine_learning/")
    model_path = str(ml_folder+'/models/')

    ## Use oh_flat to encode labels as one-hot for RandomForestClassifier
    oh_flat = True

    mlc = MLClassifier()
    model_clf, test_score = mlc.model_evaluation(model_path, model_name, X_test, y_test, oh_flat)


    # explainer = shap.Explainer(model_clf[0])
    # ## Return a class 'shap._explanation.Explanation'
    # shap_values = explainer(X_test)
    # # print('shap_values_:', shap_values)
    # print('shap_values_:', shap_values.shape)
    # print('shap_values_:', shap_values.base_values)
    #
    # # base_values = shap_values.base_values
    # # print("base_values")
    #
    # # visualize the first prediction's explanation
    # # shap.plots.waterfall(shap_values.base_values[0], values[0][0], X[0]))
    # shap.plots.waterfall(shap_values[0], shap_values.values, shap_values.data)


    ##---------------------------------------------------------
    ## Iris
    # X,y = shap.datasets.adult()
    #
    # print("X: ", X)
    # print("X: ", type(X))
    # print("X: ", X.shape)
    #
    # X["Occupation"] *= 1000 # to show the impact of feature scale on KNN predictions
    # X_display,y_display = shap.datasets.adult(display=True)
    #
    #
    # X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
    #             X_data, y_data, test_size=0.2, random_state=7)
    #
    # model_clf = sklearn.neighbors.KNeighborsClassifier()
    # model_clf.fit(X_train, y_train)
    #
    # f = lambda x: model_clf.predict_proba(x)[:,1]
    # med = X_train.median().values.reshape((1,X_train.shape[1]))
    #
    # ##---------------------------------------------------------
    #
    #
    # # f = lambda x: model_clf.predict_proba(x)[:,1]
    # # med = X_train.median().values.reshape((1,X_train.shape[1]))
    #
    # explainer = shap.Explainer(f, med)
    # shap_values = explainer(X_test.iloc[:,:])
    #
    # # shap.plots.waterfall(shap_values[0])
    #
    # shap.plots.heatmap(shap_values)
    #
    # # ## summarize the effects of all the features
    # # shap.plots.beeswarm(shap_values)
    # #
    # # shap.plots.bar(shap_values




    ###################################################################
    ###################################################################
    # train an XGBoost model
    X, y = shap.datasets.boston()

    # print("X-bostom: ", X)
    # print("y-bostom: ", y)
    #
    #
    # model = xgboost.XGBRegressor().fit(X, y)


    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    shap.initjs()

    X ,X_test, y ,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)


    rforest = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    rforest.fit(X, y)
    print(rforest.predict)


    explainer = shap.KernelExplainer(rforest.predict_proba, X)
    shap_values = explainer.shap_values(X.iloc[0,:])
    shap.force_plot(explainer.expected_value[0], shap_values[0], X.iloc[0,:])

    print("explainer.expected_value[0]: ", explainer.expected_value[0])
    print("explainer.expected_value[0]: ", explainer.expected_value[1])
    print("explainer.expected_value[0]: ", explainer.expected_value)




    # # explain the model's predictions using SHAP
    # # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    # explainer = shap.Explainer(model)
    # shap_values = explainer(X)
    #
    # # visualize the first prediction's explanation
    # shap.plots.waterfall(shap_values[1])
    #
    # # visualize the first prediction's explanation with a force plot
    # shap.initjs()
    # shap.plots.force(shap_values[0])
    #
    # # print(shap_plot)
    # soup = BeautifulSoup(shap.plots.force(shap_values[1]), 'html.parser')
    #
    # # html_string = data.show_batch().data
    #
    # print(soup.prettify())
    # shap.plots.force(shap_values[0])

    # with open("/home/jagh/Documents/01_UB/RadiologyAI-Engine/testbed-MIA/02_GENERAL/metrics_folder/shap_plot.html", "w") as file:
    #     file.write(shap_plot)






    # ###############################################################
    # ## C- Individual SHAP Value Plot
    #
    # # Get the predictions and put them with the test data.
    # X_output = X_test.copy()
    # X_output.loc[:,'predict'] = np.round(model_clf.predict(X_output),2)
    #
    # # Randomly pick some observations
    # # random_picks = np.arange(1,330,50) # Every 50 rows
    # S = X_output.iloc[3]
    #
    # print("S: ", S)
    #
    #
    # # explain all the predictions in the test set
    # explainer = shap.TreeExplainer(rforest)
    # shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values, X_test)
    #
    #
    # ##---------------------------------------
    # ## Initialize your Jupyter notebook with initjs(), otherwise you will get an error message.
    # shap.initjs()
    #
    # # explainerModel = shap.TreeExplainer(knn)
    # explainerModel = shap.Explanation(knn)
    # shap_values_Model = explainerModel.shap_values(S)
    #
    # shap.force_plot(explainerModel.expected_value, shap_values_Model[j], S.iloc[[j]])





    # ## Multiclass lesion features
    # ##-----------------------------------------------------
    # ## CON
    # shap.plots.scatter(shap_values[:,"BB_PixelSurface.1"])
    # shap.plots.scatter(shap_values[:,"BB_MeshSurface"])
    # shap.plots.scatter(shap_values[:,"AA_Kurtosis"])
    # ##-----------------------------------------------------
    # ## GGO
    # shap.plots.scatter(shap_values[:,"BB_PixelSurface"])
    # shap.plots.scatter(shap_values[:,"GG_SizeZoneNonUniformity"])
    # shap.plots.scatter(shap_values[:,"AA_Maximum"])
    # ##-----------------------------------------------------
    # ## PLE
    # shap.plots.scatter(shap_values[:,"EE_Busyness"])
    # shap.plots.scatter(shap_values[:,"AA_90Percentile"])
    # shap.plots.scatter(shap_values[:,"FF_DependenceEntropy"])
    # ##-----------------------------------------------------
    # ## BAN
    # shap.plots.scatter(shap_values[:,"CC_DifferenceEntropy"])


    # ## General-Class lesion features
    # ##-----------------------------------------------------
    # shap.plots.scatter(shap_values[:,"AA_Median"])
    # shap.plots.scatter(shap_values[:,"BB_MeshSurface"])
    # shap.plots.scatter(shap_values[:,"CC_Autocorrelation"])
    # shap.plots.scatter(shap_values[:,"CC_ClusterProminence"])
    # shap.plots.scatter(shap_values[:,"GG_GrayLevelNonUniformity"])
    # shap.plots.scatter(shap_values[:,"GG_SizeZoneNonUniformity"])


    #########################################################################
    #########################################################################
    # ## Accessing the data
    # ## Load dataset
    # radiomics_folder = os.path.join(testbed, experiment_name, "radiomics_features/")
    # # train_lesion_features_file_path = os.path.join(radiomics_folder, "general2class-Tr-FullFeatures.csv")
    # train_lesion_features_file_path = os.path.join(radiomics_folder, "cov2radiomics-Tr-FeatureSelection.csv")
    # X_train, y_train = load_features(train_lesion_features_file_path)
    # print("X_train: {} || y_train: {} ".format(str(X_train.shape), str(y_train.shape)))
    #
    #
    # # train an XGBoost model
    # model = xgboost.XGBRegressor().fit(X_train, y_train)
    #
    # # explain the model's predictions using SHAP
    # # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    # explainer = shap.Explainer(model)
    # shap_values = explainer(X_train)
    #
    # # visualize the first prediction's explanation
    # shap.plots.waterfall(shap_values[0])
    #
    # # # visualize the first prediction's explanation with a force plot
    # # shap.plots.force(shap_values[0])
    #
    # ## summarize the effects of all the features
    # shap.plots.beeswarm(shap_values)
    #
    # shap.plots.bar(shap_values)
    #
    # # create a dependence scatter plot to show the effect of a single feature across the whole dataset
    # # shap.plots.scatter(shap_values[:,], color=shap_values)
    #
    #
    # # ## Load dataset
    # # radiomics_folder = os.path.join(testbed, experiment_name, "radiomics_features/")
    # # test_lesion_features_file_path = os.path.join(radiomics_folder, experiment_filename)
    # # X_test, y_test = load_features(test_lesion_features_file_path)
    # # print("X_test: {} || y_test: {} ".format(str(X_test.shape), str(y_test.shape)))
    #
    #
    # # # plot the SHAP values for the Setosa output of all instances
    # # shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, link="logit")


from sklearn.model_selection import StratifiedShuffleSplit
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



import pickle
import xgboost as xgb

import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import f1_score

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
        xgb_model = xgb.XGBClassifier(n_jobs=1).fit(X[train_index], y[train_index])

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
        labels_name = ['3', '4', '5', '6', '8', '9']
        # labels_name = ['4', '5', '6', '8', '9']
        plot_confusion_matrix(xgb_model, X, y,
                                    display_labels=labels_name,
                                    cmap=plt.cm.Blues,
                                    # normalize='true'
                                    ) #, xticks_rotation=15)
        plt.title(str(" XGBoost || F1-score: "+ str(train_score)))
        plt.show()



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
        labels_name = ['3', '4', '5', '6', '8', '9']
        plot_confusion_matrix(xgb_model, X_test, y_test,
                                    display_labels=labels_name,
                                    cmap=plt.cm.Blues,
                                    # normalize='true'
                                    ) #, xticks_rotation=15)
        plt.title(str(" XGBoost || F1-score: "+ str(test_score)))
        plt.show()







def run(args):
    testbed = "testbed-WHO-20220325/"

    ## Features selected
    experiment_name = "03_MULTICLASS"
    # experiment_filename = "cov2radiomics-Tr-FeatureSelection-WHO.csv"
    experiment_filename = "cov2radiomics-Ts-FeatureSelection-WHO.csv"

    # ## Features selected
    # experiment_name = "02_GENERAL-FI"
    # experiment_filename = "3DgeneralclassFF_DT-Tr.csv"
    # # experiment_filename = "3DGGO-118S-FF-2T-Ts.csv"


    # ## Features selected
    # experiment_name = "02_MULTI"
    # # experiment_filename = "3DMUL-118S-FF-2T-Tr.csv"
    # experiment_filename = "3DMUL-118S-FF-2T-Ts.csv"


    # ## Features selected
    # experiment_name = "02_GGO-FI"
    # # experiment_filename = "3DGGO-118S-FF-2T-Tr.csv"
    # experiment_filename = "3DGGO-118S-FF-2T-Ts.csv"


    # ## Features selected
    # experiment_name = "02_CON"
    # # experiment_filename = "3DCON-118S-FF-2T-Tr.csv"
    # experiment_filename = "3DCON-118S-FF-2T-Ts.csv"

    # ## Features selected
    # experiment_name = "02_PLE-FI"
    # # experiment_filename = "3DPLE-118S-FF-2T-Tr.csv"
    # # experiment_filename = "3DPLE-118S-FF-2T-Ts.csv"

    # ## Features selected
    # experiment_name = "02_BAN"
    # experiment_filename = "3DBAN-118S-FF-2T-Tr.csv"
    # # experiment_filename = "3DBAN-118S-FF-2T-Ts.csv"




    # Select the model to evaluate
    model_name = 'RandomForestClassifier'
                    #'LogisticRegression'
                    #'RandomForestClassifier'
                    #'GradientBoostingClassifier'

    ## 00 - StratifiedShuffleSplit
    # stratifiedShuffleSplit(testbed, experiment_name, experiment_filename)

    ## 01 - Train
    # ml_grid_search(testbed, experiment_name, experiment_filename)

    ## 02 - Test
    model_evaluation(testbed, experiment_name, experiment_filename, model_name)

    # ## 03 -- Alternative
    # xgboostTraining(testbed, experiment_name, experiment_filename)

    ## 03 - Reliability metrics
    # model_entropy(testbed, experiment_name, experiment_filename, model_name)

    ## 04 - Feature Vsualization
    # feature_visualization(testbed, experiment_name, experiment_filename, model_name)
    # plot_radiomic_features(testbed, experiment_name, experiment_filename, model_name)

    ## 05 - Lasso ensemble for explainer
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
