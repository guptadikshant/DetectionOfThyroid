import os
from DataMgmt import model_data
from DataPreprocessing import data_preprocess
from sklearn.model_selection import train_test_split
from BestModelFinder import Tuner
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


class TrainModel:

    def __init__(self):
        """
        Initializing model and training data
        """
        self.data = model_data.get_data()
        self.preprocessor = data_preprocess.Preprocessor()

    def separate_train_test_split(self):
        """
        Preprocess the data use for training and separate
        the training data into training and validation dataset
        :return: x_train,y_train,x_valid,y_valid
        :rtype: dataframe and series
        """

        logging.info("Entering the separate_train_test_split method")

        self.preprocessor.check_null_values(self.data)
        self.train_data = self.preprocessor.encode_and_impute_data(self.data)
        self.x, self.y = self.preprocessor.separate_label_feature(self.train_data, "Class")
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.x,
                                                                                  self.y,
                                                                                  test_size=0.3,
                                                                                  random_state=0)

        logging.info("Data is split into training and validation. Exiting the function")

        return self.x_train, self.y_train, self.x_valid, self.y_valid

    def preprocess_data(self):
        """
        This function handles the training and validation data if
        it is imbalance and and use encoded and imputed pipeline
        to encode and impute the values
        :return: x_train_processed, y_train_sampled, x_valid_processed, y_valid_sampled
        :rtype: Numpy array
        """

        logging.info("Entering the preprocess_data method of training")

        self.x_train_sampled, self.y_train_sampled, self.x_valid_sampled, self.y_valid_sampled = self.preprocessor.handle_imbalance_data(
            self.x_train,
            self.y_train,
            self.x_valid,
            self.y_valid)

        self.x_train_processed, self.x_valid_processed = self.preprocessor.preprocessor_pipeline(self.x_train_sampled,
                                                                                                 self.x_valid_sampled)

        logging.info("Data is preprocessed and also handles imbalances in the training dataset. Exiting the function")

        return self.x_train_processed, self.y_train_sampled, self.x_valid_processed, self.y_valid_sampled

    def best_model_select(self):
        """
        This function checking the accuracy of the training
        and validation data and save the best model based
        on the best accuracy score
        """

        logging.info("Entering the best_model_select method.")

        tuner_obj = Tuner.ModelTuner()
        tuner_obj.get_train_accuracy(self.x_train_processed, self.y_train_sampled)
        tuner_obj.get_best_model(self.x_train_processed, self.y_train_sampled, self.x_valid_processed,
                                 self.y_valid_sampled)
        tuner_obj.save_best_model()

        logging.info("Training of the model is completed. Exiting the training module")
