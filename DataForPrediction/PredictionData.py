import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import os

import logging
os.makedirs("Application_Logs", exist_ok=True)

logging.basicConfig(
    filename=os.path.join("Application_Logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)

def data_from_data_for_pred():
    """
    this function return the data use for prediction
    :return: pred_data
    :rtype: DataFrame
    """
    logging.info("Entering the prediction data function")
    path = os.path.join("DataForPrediction", "Data For Prediction.csv")
    pred_data = pd.read_csv(path)
    logging.info("Prediction Data read fine. Exiting the prediction data function")
    return pred_data

