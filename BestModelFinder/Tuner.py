import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
import os
import pickle
import warnings

warnings.filterwarnings("ignore")
import logging
os.makedirs("Application_Logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("Application_Logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)


class ModelTuner:
    """
    This Class is used to first train different models on the train data and check the ROC_AUC scores
    on validation data. Then after comparison it finds the best model and saves it.
    """

    def __init__(self):
        """
        Intialize different models
        """
        self.lrmodel = LogisticRegression()
        self.dtmodel = DecisionTreeClassifier()
        self.etmodel = ExtraTreesClassifier()
        self.rfmodel = RandomForestClassifier()
        self.admodel = AdaBoostClassifier()
        self.gbmodel = GradientBoostingClassifier()
        self.xgmodel = XGBClassifier()
        self.knnmodel = KNeighborsClassifier()
        self.nbmodel = MultinomialNB()

    def get_best_param_logistic(self, x_train, y_train):
        """
        This function used to find the best parameters for logistic regression model, trains on it and returns it
        after training.
        :return: self.lrmodel
        :rtype: model
        """

        logging.info("Entering the best parameter function of Logistic Regression")
        print("Entering the best parameter function of Logistic Regression")
        param_logistic = {
            "penalty": ["l1", "l2", "elasticnet", "none"],
            "tol": [1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
            "C": list(np.arange(1.0, 10.0)),
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "l1_ratio": list(np.arange(0, 1, 0.1))
        }

        logistic_search = RandomizedSearchCV(self.lrmodel,
                                             param_logistic,
                                             cv=5,
                                             random_state=0,
                                             verbose=True,
                                             n_jobs=-1)

        logistic_search.fit(x_train, y_train)

        best_params = logistic_search.best_params_

        self.lrmodel = LogisticRegression(**best_params, max_iter=10000)

        self.lrmodel.fit(x_train, y_train)

        logging.info(f"the best parameter function for Logistic Regression are {best_params}. Exiting the function.")

        print("Exiting logistic regression")

        return self.lrmodel

    def get_best_param_decisiontree(self, x_train, y_train):
        """
        This function used to find the best parameters for decision tree model, trains on it and returns it
        after training.
        :return: self.dtmodel
        :rtype: model
        """

        logging.info("Entering the best parameter function of Decision Tree")
        print("Entering the best parameter function of Decision Tree")

        param_grid = {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"],
            "max_depth": range(1, 10, 1),
            "min_samples_split": range(2, 10),
            "min_samples_leaf": range(1, 5),
            "max_features": ["auto", "sqrt", "log2"],

        }

        decisiontree_search = RandomizedSearchCV(self.dtmodel,
                                                 param_grid,
                                                 cv=5,
                                                 random_state=0,
                                                 verbose=True,
                                                 n_jobs=-1)

        decisiontree_search.fit(x_train, y_train)

        best_params = decisiontree_search.best_params_

        self.dtmodel = DecisionTreeClassifier(**best_params)

        self.dtmodel.fit(x_train, y_train)

        logging.info(f"the best parameter function for Decision Tree are {best_params}.Exiting the function.")

        print("Exiting Decision Tree")

        return self.dtmodel

    def get_best_param_randomforest(self, x_train, y_train):
        """
         This function used to find the best parameters for random forest model, trains on it and returns it
         after training.
         :return: self.rfmodel
         :rtype: model
         """

        logging.info("Entering the best parameter function of Random Forest")
        print("Entering the best parameter function of Random Forest")

        param_grid = {
            "n_estimators": range(100, 1000, 100),
            "criterion": ["gini", "entropy"],
            "max_depth": range(1, 10, 1),
            "min_samples_split": range(2, 10),
            "min_samples_leaf": range(1, 5),
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": ["True", "False"]
        }

        randomforest_search = RandomizedSearchCV(self.rfmodel,
                                                 param_grid,
                                                 cv=5,
                                                 verbose=True,
                                                 n_jobs=-1)

        randomforest_search.fit(x_train, y_train)

        best_params = randomforest_search.best_params_

        self.rfmodel = RandomForestClassifier(**best_params)

        self.rfmodel.fit(x_train, y_train)

        logging.info(f"the best parameter function for Random Forest are {best_params}.Exiting the function.")

        print("Exiting Random Forest")

        return self.rfmodel

    def get_best_param_extratree(self, x_train, y_train):
        """
         This function used to find the best parameters for extra tree model, trains on it and returns it
         after training.
         :return: self.etmodel
         :rtype: model
         """

        logging.info("Entering the best parameter function of Extra Tree")
        print("Entering the best parameter function of Extra Tree")

        param_grid = {
            "n_estimators": range(100, 1000, 100),
            "criterion": ["gini", "entropy"],
            "max_depth": range(1, 10, 1),
            "min_samples_split": range(2, 10),
            "min_samples_leaf": range(1, 5),
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": ["True", "False"]
        }

        extratree_search = RandomizedSearchCV(self.rfmodel,
                                              param_grid,
                                              cv=5,
                                              verbose=True,
                                              n_jobs=-1)

        extratree_search.fit(x_train, y_train)

        best_params = extratree_search.best_params_

        self.etmodel = ExtraTreesClassifier(**best_params)

        self.etmodel.fit(x_train, y_train)

        logging.info(f"the best parameter function for Extra Tree are {best_params}.Exiting the function.")

        print("Exiting Extra Tree function")

        return self.etmodel

    def get_best_param_adaboost(self, x_train, y_train):
        """
         This function used to find the best parameters for ada boost model, trains on it and returns it
         after training.
         :return: self.admodel
         :rtype: model
         """

        logging.info("Entering the best parameter function of Ada Boost")

        print("Entering the best parameter function of Ada Boost")

        param_grid = {
            "base_estimator": [DecisionTreeClassifier(), RandomForestClassifier()],
            "n_estimators": range(100, 1000, 100),
            "learning_rate": list(np.arange(1.0, 10.0, 1.0)),
            "algorithm": ["SAMME", "SAMME.R"]
        }

        adaboost_search = RandomizedSearchCV(self.admodel,
                                             param_grid,
                                             cv=5,
                                             verbose=True,
                                             n_jobs=-1)

        adaboost_search.fit(x_train, y_train)

        best_params = adaboost_search.best_params_

        self.admodel = AdaBoostClassifier(**best_params)

        self.admodel.fit(x_train, y_train)

        logging.info(f"the best parameter function for Ada Boost are {best_params}.Exiting the function.")

        print("Exiting Ada boost function")

        return self.admodel

    def get_best_param_gbboost(self, x_train, y_train):
        """
         This function used to find the best parameters for gradient boosting model, trains on it and returns it
         after training.
         :return: self.gbmodel
         :rtype: model
         """

        logging.info("Entering the best parameter function of Gradient Boosting")

        print("Entering the best parameter function of Gradient Boosting")

        param_grid = {
            "loss": ["deviance", "exponential"],
            "learning_rate": list(np.arange(1.0, 10.0, 1.0)),
            "n_estimators": range(100, 1000, 100),
            "subsample": list(np.arange(0.1, 0.9, 0.1)),
            "criterion": ["friedman_mse", "squared_error"],
            "min_samples_split": range(2, 10),
            "min_samples_leaf": range(1, 5),
            "max_features": ["auto", "sqrt", "log2"],
            "max_depth": range(1, 10, 1),
            "min_impurity_decrease": list(np.arange(1.0, 10.0)),
            "tol": [1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
        }

        gbboost_search = RandomizedSearchCV(self.gbmodel,
                                            param_grid,
                                            cv=5,
                                            verbose=True,
                                            n_jobs=-1)

        gbboost_search.fit(x_train, y_train)

        best_params = gbboost_search.best_params_

        self.gbmodel = GradientBoostingClassifier(**best_params)

        self.gbmodel.fit(x_train, y_train)

        logging.info(f"the best parameter function for Gradient Boosting are {best_params}.Exiting the function.")

        print("Exiting Gradient Boosting function")

        return self.gbmodel

    def get_best_param_xgboost(self, x_train, y_train):
        """
         This function used to find the best parameters for xgboost classifier model, trains on it and returns it
         after training.
         :return: self.xgmodel
         :rtype: model
         """

        logging.info("Entering the best parameter function of XG Boost")

        print("Entering the best parameter function of XG Boost")

        param_grid = {
            "n_estimators": range(100, 1000, 100),
            "max_depth": range(1, 10, 1),
            "learning_rate": list(np.arange(1.0, 10.0, 1.0)),
            "gamma": range(0, 10, 1),
            "min_child_weight": range(1, 10, 1),
            "max_delta_step": range(0, 10, 1),
            "subsample": list(np.arange(0.1, 0.9, 0.1)),
            "tree_method": ["auto", "exact", "approx", "hist", "gpu_hist"],
            "max_leaves": range(0, 5, 1),
            "predictor": ["auto", "cpu_predictor", "gpu_predictor"],

        }

        xgboost_search = RandomizedSearchCV(self.xgmodel,
                                            param_grid,
                                            cv=5,
                                            n_jobs=-1,
                                            verbose=True)

        xgboost_search.fit(x_train, y_train)

        best_params = xgboost_search.best_params_

        self.xgmodel = XGBClassifier(**best_params)

        self.xgmodel.fit(x_train, y_train)

        logging.info(f"the best parameter function for XG Boost are {best_params}.Exiting the function.")

        print("Exiting XGBoost function")

        return self.xgmodel

    def get_best_param_knn(self, x_train, y_train):
        """
         This function used to find the best parameters for KNN model, trains on it and returns it
         after training.
         :return: self.knnmodel
         :rtype: model
         """

        logging.info("Entering the best parameter function of KNN")

        print("Entering the best parameter function of KNN")

        param_grid = {
            "n_neighbors": range(1, 20, 3),
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": range(10, 100, 10),
            "p": [1, 2]
        }

        knn_search = RandomizedSearchCV(self.knnmodel,
                                        param_grid,
                                        cv=5,
                                        verbose=True,
                                        n_jobs=-1)

        knn_search.fit(x_train, y_train)

        best_params = knn_search.best_params_

        self.knnmodel = KNeighborsClassifier(**best_params)

        self.knnmodel.fit(x_train, y_train)

        logging.info(f"the best parameter function for KNN are {best_params}.Exiting the function.")

        print("Exiting KNN function")

        return self.knnmodel

    def get_best_param_nbayes(self, x_train, y_train):
        """
         This function used to find the best parameters for naive bayes model, trains on it and returns it
         after training.
         :return: self.nbmodel
         :rtype: model
         """

        logging.info("Entering the best parameter function of Naive Bayes")

        print("Entering the best parameter function of Naive Bayes")

        param_grid = {
            "alpha": list(np.arange(1.0, 10.0)),
            "fit_prior": ["True", "False"]
        }

        nbayes_search = RandomizedSearchCV(self.nbmodel,
                                           param_grid,
                                           cv=5,
                                           verbose=True,
                                           n_jobs=-1)

        nbayes_search.fit(x_train, y_train)

        best_params = nbayes_search.best_params_

        self.nbmodel = MultinomialNB(**best_params)

        self.nbmodel.fit(x_train, y_train)

        logging.info(f"the best parameter function for Naive Bayes are {best_params}.Exiting the function.")

        print("Exiting Naive Bayes function")

        return self.nbmodel

    def get_train_accuracy(self, x_train, y_train):
        """
        This function is used to get the train accuracy of different models.
        :return: Train Accuracy ROC Score
        :rtype: Tuple
        """

        logging.info("Entering the function of train accuracy")

        print("Entering train accuracy function")

        self.lrmodel_trained = self.get_best_param_logistic(x_train, y_train)
        self.dtmodel_trained = self.get_best_param_decisiontree(x_train, y_train)
        self.rfmodel_trained = self.get_best_param_randomforest(x_train, y_train)
        self.etmodel_trained = self.get_best_param_extratree(x_train, y_train)
        self.admodel_trained = self.get_best_param_adaboost(x_train, y_train)
        self.gbmodel_trained = self.get_best_param_gbboost(x_train, y_train)
        self.xgmodel_trained = self.get_best_param_xgboost(x_train, y_train)
        self.knnmodel_trained = self.get_best_param_knn(x_train, y_train)
        self.nbmodel_trained = self.get_best_param_nbayes(x_train, y_train)

        logistic_pred = self.lrmodel_trained.predict(x_train)
        decision_pred = self.dtmodel_trained.predict(x_train)
        random_pred = self.rfmodel_trained.predict(x_train)
        extratree_pred = self.etmodel_trained.predict(x_train)
        adaboost_pred = self.admodel_trained.predict(x_train)
        gbboost_pred = self.gbmodel_trained.predict(x_train)
        xgboost_pred = self.xgmodel_trained.predict(x_train)
        knn_pred = self.knnmodel_trained.predict(x_train)
        nbayes_pred = self.nbmodel_trained.predict(x_train)

        logistic_rocscore = accuracy_score(y_train, logistic_pred)
        decisiontree_rocscore = accuracy_score(y_train, decision_pred)
        randomforest_rocscore = accuracy_score(y_train, random_pred)
        extratree_rocscore = accuracy_score(y_train, extratree_pred)
        adaboost_rocscore = accuracy_score(y_train, adaboost_pred)
        gbboost_rocscore = accuracy_score(y_train, gbboost_pred)
        xgbbost_rocscore = accuracy_score(y_train, xgboost_pred)
        knn_rocscore = accuracy_score(y_train, knn_pred)
        nbayes_rocscore = accuracy_score(y_train, nbayes_pred)

        trainacc = [logistic_rocscore, decisiontree_rocscore, randomforest_rocscore,
                    extratree_rocscore, adaboost_rocscore, gbboost_rocscore, xgbbost_rocscore,
                    knn_rocscore, nbayes_rocscore]

        logging.info(f"The train accuracy of different models is \n {trainacc}. \n Exiting the function.")

        print("Exiting train accuracy function")

    def get_best_model(self, x_train, y_train, x_valid, y_valid):
        """
        This function is used to get the details of the best model.
        :return: model name
         """

        logging.info("Entering the function of validation accuracy")

        print("Entering the function of validation accuracy")

        logistic_pred = self.lrmodel_trained.predict(x_valid)
        decision_pred = self.dtmodel_trained.predict(x_valid)
        random_pred = self.rfmodel_trained.predict(x_valid)
        extratree_pred = self.etmodel_trained.predict(x_valid)
        adaboost_pred = self.admodel_trained.predict(x_valid)
        gbboost_pred = self.gbmodel_trained.predict(x_valid)
        xgboost_pred = self.xgmodel_trained.predict(x_valid)
        knn_pred = self.knnmodel_trained.predict(x_valid)
        nbayes_pred = self.nbmodel_trained.predict(x_valid)

        logistic_rocscore = accuracy_score(y_valid, logistic_pred)
        decisiontree_rocscore = accuracy_score(y_valid, decision_pred)
        randomforest_rocscore = accuracy_score(y_valid, random_pred)
        extratree_rocscore = accuracy_score(y_valid, extratree_pred)
        adaboost_rocscore = accuracy_score(y_valid, adaboost_pred)
        gbboost_rocscore = accuracy_score(y_valid, gbboost_pred)
        xgbbost_rocscore = accuracy_score(y_valid, xgboost_pred)
        knn_rocscore = accuracy_score(y_valid, knn_pred)
        nbayes_rocscore = accuracy_score(y_valid, nbayes_pred)

        validation_acc = [logistic_rocscore, decisiontree_rocscore, randomforest_rocscore,
                          extratree_rocscore, adaboost_rocscore, gbboost_rocscore, xgbbost_rocscore,
                          knn_rocscore, nbayes_rocscore]

        logging.info(f"The validation accuracy of different models is \n {validation_acc}.")

        best_model_dict = {
            self.lrmodel: logistic_rocscore,
            self.dtmodel: decisiontree_rocscore,
            self.rfmodel: randomforest_rocscore,
            self.etmodel: extratree_rocscore,
            self.admodel: adaboost_rocscore,
            self.gbmodel: gbboost_rocscore,
            self.xgmodel: xgbbost_rocscore,
            self.knnmodel: knn_rocscore,
            self.nbmodel: nbayes_rocscore
        }
        # To find the best model.
        self.best_model = max(zip(best_model_dict.values(), best_model_dict.keys()))[1]

        logging.info(f"The best model is \n {self.best_model}.\n Exiting the function")

        print("Exiting validation accuracy function")

        return self.best_model

    def save_best_model(self):
        """
        This function saves the best model in the BestModel Folder
        :return:BestModel
        :rtype:Model.pkl
        """

        logging.info("Entering the function of save best model")

        print("entering save model function")

        modelname = self.best_model
        model_dir = "BestModel"
        os.makedirs(model_dir, exist_ok=True)
        filename = "Model.pkl"
        model_path = os.path.join(model_dir, filename)
        pickle.dump(modelname, open(model_path, "wb"))

        logging.info(f"The best model is saved at {model_path}. Exiting the module")

        print("exiting save model function")
