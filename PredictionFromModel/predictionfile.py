from DataForPrediction import PredictionData
from DataPreprocessing import PredictionDataPreprocess
from Application_Logging import logging
import pickle
import warnings
warnings.filterwarnings("ignore")

class PredictionModel:

    def __init__(self):
        """
        Initializing the model and the preprocessing process pipeline
        """
        self.preprocessor = PredictionDataPreprocess.Preprocessor()
        self.prediction_data = PredictionData.data_from_data_for_pred()
        self.loaded_model = pickle.load(open(r"C:\Users\Dikshant\Downloads\New folder\Thyroid-Detection-Project\BestModel\Model.pkl", "rb"))
        self.file_object = open("Application_Logging/Prediction_From_Model.txt", 'a+')
        self.log_writer = logging.App_Logger()

    def preprocess_pred_data(self):
        """
        Preprocess the data use for prediction and return a dataset having encoded
        and imputed values
        :return: final_pred_data
        :rtype: DataFrame
        """
        self.log_writer.log(self.file_object,"Entering the prediction file module ")
        self.preprocessor.check_null_values(self.prediction_data)
        preprocessed_pred_data = self.preprocessor.encode_and_impute_data(self.prediction_data)
        self.final_pred_data = self.preprocessor.preprocessor_pipeline(preprocessed_pred_data)
        self.log_writer.log(self.file_object,"Preprocessing of prediction data completed")
        return self.final_pred_data


    def make_prediction(self):
        """
        Making prediction and saving it in a list to store in database
        show to user
        :return: final_pred
        :rtype: list
        """
        self.log_writer.log(self.file_object,"Entering the make prediction function")
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

        self.log_writer.log(self.file_object, "All the prediction are done and stored in final_pred list")

        return final_pred



