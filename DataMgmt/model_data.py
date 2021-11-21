import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import os

import logging

logging.basicConfig(
    filename=os.path.join("Application_Logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)

def get_data():
    """
    This function manages the input data
    :return: df
    """
    logging.info("Entering the train data function")
    path = os.path.join("DataMgmt", "thyroid_data.csv")
    df = pd.read_csv(path)
    logging.info("Training Data read fine. Exiting the training data function")
    return df
