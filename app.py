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
    return render_template("home.html")


@app.route('/prediction',methods=['GET', 'POST'])
def pred_route():
    pred_obj = predictionfile.PredictionModel()
    pred_obj.preprocess_pred_data()
    pred = pred_obj.make_prediction()
    # pred_obj.savemodelprediction()
    return render_template("predict.html", pred = pred[:5])


if __name__ == '__main__':
    app.run(debug=True)
