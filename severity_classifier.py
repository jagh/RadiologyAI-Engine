
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








def run(args):
    testbed = "testbed-ECR22/"

    # ## Features selected
    experiment_name = "02_GENERAL"
    experiment_filename = "general2class-Ts-FeatureSelection-2.csv"

    ## Features selected by each lesion
    # experiment_name = "03_MULTICLASS"
    # experiment_filename = "cov2radiomics-Tr-FeatureSelection.csv"

    # Select the model to evaluate
    model_name = 'LogisticRegression'
                    #'LogisticRegression'
                    #'RandomForestClassifier'
                    #'GradientBoostingClassifier'

    # ml_grid_search(testbed, experiment_name, experiment_filename)
    model_evaluation(testbed, experiment_name, experiment_filename, model_name)

    model_entropy(testbed, experiment_name, experiment_filename, model_name)



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
