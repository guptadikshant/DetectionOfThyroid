import os
from DataForPrediction import PredictionData
from DataPreprocessing import PredictionDataPreprocess
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

class PredictionModel:

    def __init__(self):
        """
        Initializing the model and the preprocessing process pipeline
        """
        self.preprocessor = PredictionDataPreprocess.Preprocessor()
        self.prediction_data = PredictionData.data_from_data_for_pred()
        self.path = os.path.join("BestModel","Model.pkl")
        self.loaded_model = pickle.load(open( self.path, "rb"))

    def preprocess_pred_data(self):
        """
        Preprocess the data use for prediction and return a dataset having encoded
        and imputed values
        :return: final_pred_data
        :rtype: DataFrame
        """
        logging.info("Entering the prediction file module ")

        self.preprocessor.check_null_values(self.prediction_data)
        preprocessed_pred_data = self.preprocessor.encode_and_impute_data(self.prediction_data)
        self.final_pred_data = self.preprocessor.preprocessor_pipeline(preprocessed_pred_data)

        logging.info("Preprocessing of prediction data completed")

        return self.final_pred_data


    def make_prediction(self):
        """
        Making prediction and saving it in a list to store in database
        show to user
        :return: final_pred
        :rtype: list
        """
        logging.info("Entering the make prediction function")

        final_pred = []
        
        y_pred = self.loaded_model.predict(self.final_pred_data)

        for ele in y_pred:
            if ele == 0:
                ele = "negative"
            elif ele == 1:
                ele = "compensated_hypothyroid"
            elif ele == 2:
                ele = "primary_hypothyroid"
            else:
                ele = "secondary_hypothyroid"
            final_pred.append(ele)

        logging.info("All the prediction are done and stored in final_pred list")

        return final_pred



