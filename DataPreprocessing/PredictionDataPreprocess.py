import os
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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


class Preprocessor:

    def __init__(self):
        pass

    def check_null_values(self, data):

        """
        Checking the null values in the Prediction Data and storing
        the features in the PredictionDataWithMissingValues.CSV file
        """

        logging.info("Start of the Prediction Data Preprocessing")

        feature_with_null = [feature for feature in data.columns if data[feature].isnull().sum() > 0]

        if len(feature_with_null) > 0:

            dataframe_with_null = data[feature_with_null].isnull().sum().to_frame().reset_index()
            dataframe_with_null.columns = ["Feature Name", "Number of Missing Values"]

            Missing_Values = "MissingValues"
            os.makedirs(Missing_Values, exist_ok=True)
            dataframe_with_null.to_csv("MissingValues/PredictionDataWithMissingValues.CSV", index=False)

            logging.info("Prediction data has some missing values. Please check the Missing value folder for more info")

        else:
            logging.info("No missing value in any feature in Prediction Data")

    def encode_and_impute_data(self, data):

        """
        Encoding some of the features of the prediction data and
        dividing the features into numerical and categorical
        :return: prediction data
        :rtype: DataFrame
        """

        logging.info("Entered encode_and_impute_data method for prediction")

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

        logging.info("Some features are encoded fine and features are divided into numerical and categorcial")

        return data

    def preprocessor_pipeline(self, data):

        """
        Creating a pipeline to encode and impute the features of the prediction data
        :return: data
        :rtype: Numpy Array
        """

        logging.info("Entered the preprocessor_pipeline method of Prediction")

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

        logging.info("All features are encoded and imputed fine. Exiting the Prediction Data Process module")

        return data
