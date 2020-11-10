import pandas as pd
import numpy as np
import pickle
import time

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from engine.utils import Utils

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class MLClassifier:
    """
    A Machine Learning Model Search Classifier Module
    """

    def __init__(self):
        pass

    def splitting(self, X_data, y_data, ml_folder):
        ## Create a ML folder and splitting the dataset
        eval_split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
        for train_index, test_index in eval_split.split(X_data, y_data):
            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            #print("train_index: {} || test_index: {} ".format(str(train_index.shape), str(test_index.shape) ))
        print("X_train: {} || y_train: {} ".format(str(X_train.shape), str(y_train.shape)))
        print("X_test: {} || y_test: {} ".format(str(X_test.shape), str(y_test.shape) ))

        ## Define location to write the split dataset
        dataset_path = str(ml_folder+'/dataset/')
        Utils().mkdir(dataset_path)

        ## Write the metadata file in txt
        np.save(str(dataset_path+"/X_train_baseline"), X_train)
        np.save(str(dataset_path+"/y_train_baseline"), y_train)
        np.save(str(dataset_path+"/X_test_baseline"), X_test)
        np.save(str(dataset_path+"/y_test_baseline"), y_test)

    def model_training(self, pipe_clf, train_x, train_y, model_path):
        """
        """

        ## Model training
        pipe_clf.fit(train_x, train_y)

        ## Writing the developing model
        model_file = open(str(model_path+"/dev_model.pkl"), "wb")
        pickle.dump(pipe_clf, model_file)
        model_file.close()

        ## Training performance
        train_preditions = pipe_clf.predict(train_x)
        train_score = f1_score(train_y, train_preditions, average='macro')
        # print("++ Train F1-Score: {}".format(train_score))

        return train_score, pipe_clf

    def model_testing(self, pipe_clf, valid_x, valid_y, model_path, oh_flat):
        """
        """
        ## Open Writing the developing model
        model_file = open(str(model_path+"/dev_model.pkl"), "rb")
        docs_clf = pickle.load(model_file, encoding='bytes')
        model_file.close()

        ## Validation performance
        valid_preditions = docs_clf.predict(valid_x)
        valid_score = f1_score(valid_y, valid_preditions, average='macro')
        # print("++ Valid F1-Score: {}".format(valid_score))

        ## Computing the confusion matrix
        # oh_flat = False
        if oh_flat == True:
            cm = confusion_matrix(valid_y.argmax(axis=1), valid_preditions.argmax(axis=1))
            # print("++ Confusion matrix: \n {}".format(cm))
        else:
            cm = confusion_matrix(valid_y, valid_preditions)
            # print("++ Confusion matrix: \n {}".format(cm))
        return valid_score

    def gridSearch(self, classifiers, X, y, oh_flat, n_splits, model_path):
        """
        ML Pipeline: Pipeline([('clf', classifier)])
        """
        train_scores = dict()
        valid_scores = dict()
        for classifier in classifiers:
            ## Machine learning Workflow
            pipe_clf = Pipeline([ ('clf', classifier) ])
            model_name = classifier.__class__.__name__
            print("--"*20)
            print("++ clf: {}".format(model_name))

            ## Encode labes as one-hot
            if model_name is 'RandomForestClassifier' and oh_flat is True:
                labels_name = ['CT-0', 'CT-1', 'CT-2', 'CT-3']
                y = pd.get_dummies(data=y, columns=labels_name).values
            else:
                oh_flat = False

            test_sizes = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10]
            for test_size in test_sizes:
                """
                Train each model over different test sizes to split the dataset
                """
                list_train_scores = list()
                list_valid_scores = list()

                this_cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
                for train_index, test_index in this_cv.split(X, y):
                    ## execution_time
                    execution_start = time.time()

                    ## Get the stratified shuffle sets by train and valid
                    train_x, valid_x = X[train_index], X[test_index]
                    train_y, valid_y = y[train_index], y[test_index]

                    ## Training and validation
                    train_score, pipe_clf = self.model_training(pipe_clf, train_x, train_y, model_path)
                    valid_score = self.model_testing(pipe_clf, valid_x, valid_y, model_path, oh_flat)

                    ## Filling the scores list to compute the mean and std
                    list_train_scores.append(train_score)
                    list_valid_scores.append(valid_score)

                    ## Model selection to write the best model by each ML classifierg
                    if not valid_scores.get(model_name):
                        if valid_score >= max(list_valid_scores):
                            model_file = open(str(model_path+"/dev_model.pkl"), "wb")
                            pickle.dump(pipe_clf, model_file)
                            model_file.close()

                    ## execution_time
                    print("Execution Time: {}".format((time.time()-execution_start)))

                ## Inserting train and valid scores
                train_scores.setdefault(model_name,[]).append(list_train_scores)
                valid_scores.setdefault(model_name,[]).append(list_valid_scores)

                ## Model selection to write the best model by each ML classifierg
                if valid_score >= np.max(valid_scores.get(model_name)):
                    model_file = open(str(model_path+"/model_"+model_name+".pkl"), "wb")
                    pickle.dump(pipe_clf, model_file)
                    model_file.close()

        return train_scores, valid_scores

    def plot_learning_curves(self, train_scores, test_scores, n_splits):
        """
        Generate the training and validation learning curves plots
        """
        print("--"*20)
        print("++ plotting the learning curves ")

        models = list(train_scores.keys())
        train_sizes = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]

        ## Set subplots
        fig, axes = plt.subplots(1, len(models), figsize=(12, 4))
        fig.tight_layout()

        for model in models:
            # print("train_scores.get(model): ", train_scores.get(model))
            m_index = models.index(model)
            train_scores_mean = np.mean(train_scores.get(model), axis=1)
            train_scores_std = np.std(train_scores.get(model), axis=1)
            test_scores_mean = np.mean(test_scores.get(model), axis=1)
            test_scores_std = np.std(test_scores.get(model), axis=1)

            ## Plot learning curve
            axes[m_index].grid()
            axes[m_index].set_title(model)
            axes[m_index].set_xlabel("Training examples")
            axes[m_index].set_ylabel("F1-Score")
            axes[m_index].set_ylim(ymin=0.0, ymax=1.0)
            # axes[m_index].set_yscale('log')

            axes[m_index].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1, color="darkturquoise")
            axes[m_index].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="mediumpurple")
            axes[m_index].plot(train_sizes, train_scores_mean, 'o-', color="darkturquoise", label="Training score")
            axes[m_index].plot(train_sizes, test_scores_mean, 'o-', color="mediumpurple", label="Validation score")
            axes[m_index].legend(loc="best")

        ## Plotting the learning curves
        plt.show()

    def model_evaluation(self, model_path, model_name, X, y, oh_flat):
        """
        ML evaluation function
        """

        model_file = open(str(model_path+'/model_'+model_name+'.pkl'), "rb")
        page_clf = pickle.load(model_file, encoding='bytes')
        model_file.close()

        predicted = page_clf.predict(X)

        ## Encode labes as one-hot
        if model_name is 'RandomForestClassifier' and oh_flat is True:
            labels_name = ['CT-0', 'CT-1', 'CT-2', 'CT-3']
            y = pd.get_dummies(data=y, columns=labels_name).values
            predicted = pd.get_dummies(data=predicted, columns=labels_name).values
        else:
            oh_flat = False

        ## Confusion matrix from binarize mabels
        if oh_flat == True:
            confusion_m = confusion_matrix(y.argmax(axis=1), predicted.argmax(axis=1))
            test_score = np.round_(f1_score(y.argmax(axis=1), predicted.argmax(axis=1),
                                                    average='macro'), decimals=3)
            print("++ Confusion matrix: \n {}".format(confusion_m))
            print("++ Performance score: {}".format(test_score))
        else:
            confusion_m = confusion_matrix(y, predicted)
            test_score = np.round_(f1_score(y, predicted, average='macro'), decimals=3)
            print("++ Confusion matrix: \n {}".format(confusion_m))
            print("++ Performance score: {}".format(test_score))

        return page_clf, test_score #, predicted
