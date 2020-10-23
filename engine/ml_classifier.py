import pandas as pd
import numpy as np
import pickle
import time


## Machine learning
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class MLClassifier:
    """
    A Machine Learning Model Search Classifier Module
    """

    def __init__(self):
        pass

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


    def model_testing(self, pipe_clf, valid_x, valid_y, model_path):
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
        oh_flat = False
        if oh_flat == True:
            cm = confusion_matrix(valid_y.argmax(axis=1), valid_preditions.argmax(axis=1))
            # print("++ Confusion matrix: \n {}".format(cm))
        else:
            cm = confusion_matrix(valid_y, valid_preditions)
            # print("++ Confusion matrix: \n {}".format(cm))

        return valid_score


    def gridSearch(self, classifiers, X, y, oh_flat, n_splits, model_path):
        """
        Machine learning Workflow:
        Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', classifier) ])
        """
        ## Binarize labels in a one-vs-all fashion
        if oh_flat == True:
            one_hot_encoding = LabelBinarizer()
            y = one_hot_encoding.fit_transform(y)
            # print("enc: ", one_hot_encoding.classes_)

            ## Write one hot encoding
            oh_file = open(oh_path, "wb")
            pickle.dump(one_hot_encoding, oh_file)
            oh_file.close()
        else:
            pass

        train_scores = dict()
        valid_scores = dict()
        for classifier in classifiers:
            ## Machine learning Workflow
            pipe_clf = Pipeline([ ('clf', classifier) ])
            model_name = classifier.__class__.__name__
            print("--"*20)
            print("++ clf: {}".format(model_name))

            test_sizes = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10]
            # train_sizes = [1417, 1526, 1635, 1744, 1853, 1962]
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
                    valid_score = self.model_testing(pipe_clf, valid_x, valid_y, model_path)

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
                             train_scores_mean + train_scores_std, alpha=0.1, color="r")
            axes[m_index].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
            axes[m_index].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            axes[m_index].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation score")
            axes[m_index].legend(loc="best")

        ## Plotting the learning curves
        plt.show()
