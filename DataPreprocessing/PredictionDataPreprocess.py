import os
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from Application_Logging import logging
import warnings
warnings.filterwarnings("ignore")


class Preprocessor:

    def __init__(self):
        self.file_object = open("Application_Logging/Prediction_Preprocessing_Logs.txt", 'a+')
        self.log_writer = logging.App_Logger()

    def check_null_values(self, data):

        self.log_writer.log(self.file_object,"Start of the Prediction Data Preprocessing")

        feature_with_null = [feature for feature in data.columns if data[feature].isnull().sum() > 0]

        if len(feature_with_null) > 0:

            dataframe_with_null = data[feature_with_null].isnull().sum().to_frame().reset_index()
            dataframe_with_null.columns = ["Feature Name", "Number of Missing Values"]

            Missing_Values = "MissingValues"
            os.makedirs(Missing_Values, exist_ok=True)
            dataframe_with_null.to_csv("MissingValues/PredictionDataWithMissingValues.CSV", index=False)

            self.log_writer.log(self.file_object , "Prediction data has some missing values. Please check the Missing value folder for more info")

        else:
            self.log_writer.log(self.file_object , "No missing value in any feature in Prediction Data")

    def encode_and_impute_data(self, data):

        data["sex"] = np.where(data["sex"] == "F", 0, 1)
        data["referral_source"] = data["referral_source"].map(
            {"other": 0, "SVI": 1, "SVHC": 2, "STMW": 4, "SVHD": 5})
        data["Class"] = data["Class"].map({"negative": 0
                                                        , "compensated_hypothyroid": 1
                                                        , "primary_hypothyroid": 2
                                                        , "secondary_hypothyroid": 3})
        self.categorical_features = [feature for feature in data.columns if len(data[feature].unique()) < 10
                                     and feature not in ["Class", "sex", "referral_source","kfold"]]

        self.numerical_features = [feature for feature in data.columns if feature not in self.categorical_features
                                   and feature not in ["Class","kfold"]]

        self.log_writer.log(self.file_object, "Some features are encoded fine and features are divided into numerical and categorcial")

        return data

    def preprocessor_pipeline(self, data):

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

        data = preprocessor.fit_transform(data)

        self.log_writer.log(self.file_object, "All features are encoded and imputed fine. Exiting the Prediction Data Process module")

        return data
