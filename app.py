from TrainingModels import training
from PredictionFromModel import predictionfile
from flask import Flask, render_template, request
import os
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if not os.path.isdir("DataForPrediction"):
            os.makedirs("DataForPrediction")
        filepath = os.path.join("DataForPrediction", file.filename)
        file.save(filepath)
        return render_template("thankyou.html")
    return render_template("home.html")


@app.route('/train', methods=['GET', 'POST'])
def train_route():
    train_obj = training.TrainModel()
    train_obj.separate_train_test_split()
    train_obj.preprocess_data()
    train_obj.best_model_select()
    return render_template("train.html")


@app.route('/prediction',methods=['GET', 'POST'])
def pred_route():
    pred_obj = predictionfile.PredictionModel()
    pred_obj.preprocess_pred_data()
    pred = pred_obj.make_prediction()
    return render_template("predict.html",pred = pred)


if __name__ == '__main__':
    app.run(debug=True)
