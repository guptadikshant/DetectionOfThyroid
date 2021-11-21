import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler
import warnings

warnings.filterwarnings("ignore")

import logging

logging.basicConfig(
    filename=os.path.join("Application_Logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)


class Preprocessor:

    def __init__(self):
        """
        Initializing the log file object
        """
        pass

    def check_null_values(self, data):

        """
        Checking the null values in the Training Data and storing
        the features in the DataWithMissingValues.CSV
        """

        feature_with_null = [feature for feature in data.columns if data[feature].isnull().sum() > 0]
        logging.info("Start of Train Data Preprocessing. checking the null values of the training data")

        if len(feature_with_null) > 0:

            dataframe_with_null = data[feature_with_null].isnull().sum().to_frame().reset_index()
            dataframe_with_null.columns = ["Feature Name", "Number of Missing Values"]

            Missing_Values = "MissingValues"
            os.makedirs(Missing_Values, exist_ok=True)
            dataframe_with_null.to_csv("MissingValues/DataWithMissingValues.CSV", index=False)
            logging.info("Feature Have Some Missing Values.Check Missing Values folder for features having missing values.Exiting the function")

        else:
            logging.info("No Missing Values in any feature.Exiting the function")

    def encode_and_impute_data(self, data):

        """
        Encoding some of the features of the training data and
        dividing the features into numerical and categorical
        :return: training data
        :rtype: DataFrame
        """

        logging.info("Entered the Encode and Impute method function of training")
        data["sex"] = np.where(data["sex"] == "F", 0, 1)
        data["referral_source"] = data["referral_source"].map({"other": 0, "SVI": 1, "SVHC": 2, "STMW": 4, "SVHD": 5})
        data["Class"] = data["Class"].map({"negative": 0
                                              , "compensated_hypothyroid": 1
                                              , "primary_hypothyroid": 2
                                              , "secondary_hypothyroid": 3})
        self.categorical_features = [feature for feature in data.columns if len(data[feature].unique()) < 10
                                     and feature not in ["Class", "sex", "referral_source"]]

        self.numerical_features = [feature for feature in data.columns if feature not in self.categorical_features
                                   and feature not in ["Class"]]

        logging.info("Exited the Encode and Impute method of training")

        return data

    def separate_label_feature(self, data, label_name):

        """
        Separating features into dependent and independent features
        :return: X,y
        :rtype: DataFrame, Series
        """
        logging.info("Entered the separate_label_feature function of training")
        self.X = data.drop(label_name, axis=1)
        self.y = data[label_name]
        logging.info("Labels are separated into dependent and independent.Exiting the function of training")
        return self.X, self.y

    def handle_imbalance_data(self, x_train, y_train, x_valid, y_valid):

        """
        Handling the imbalanceness of the training data
        :return: x_train_sampled, y_train_sampled, x_valid_sampled, y_valid_sampled
        :rtype: DataFrame and Series
        """
        logging.info("Entered the handle_imbalance_data function of training")
        rdsample = RandomOverSampler()

        x_train_sampled, y_train_sampled = rdsample.fit_resample(x_train, y_train)
        x_valid_sampled, y_valid_sampled = rdsample.fit_resample(x_valid, y_valid)

        logging.info("Balancing of data is done.Exiting the function of training")

        return x_train_sampled, y_train_sampled, x_valid_sampled, y_valid_sampled

    def preprocessor_pipeline(self, x_train_sampled, x_valid_sampled):

        """
        Creating a pipeline to encode and impute the features of the training data
        :return: x_train_processed, x_valid_processed
        :rtype: Numpy Array
        """
        logging.info("Entered the preprocessor_pipeline function of training ")
        numerical_transformer = KNNImputer(n_neighbors=2, weights='uniform', missing_values=np.nan)
        categorical_transformer = Pipeline(steps=[
            ('encoder', OrdinalEncoder()),
            ('imputer', KNNImputer(n_neighbors=2, weights='uniform', missing_values=np.nan))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        x_train_processed = preprocessor.fit_transform(x_train_sampled)
        x_valid_processed = preprocessor.transform(x_valid_sampled)

        x_train_processed = pd.DataFrame(x_train_processed, columns=x_train_sampled.columns)
        x_valid_processed = pd.DataFrame(x_valid_processed, columns=x_valid_sampled.columns)

        logging.info("All the features are encoded and imputed of training data. Exiting the module")

        return x_train_processed, x_valid_processed
