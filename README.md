
## Thyroid Detection
The Thyroid gland is a vascular gland and one of the most important organs of a human body. 
This gland secretes two hormones which help in controlling the metabolism of the body. 
The type of thyroid disorders are Compensated hypothyroid, Primary hypothyroid and Secondary hypothyroid
When this disorder occurs in the body, they release certain type of hormones into the body which imbalances the body’s metabolism. 
Thyroid related Blood test is used to detect this disease but it is often blurred and noise will be present. 
Data cleansing methods were used to make the data primitive enough for the analytics to show the risk of patients getting this disease. 
Machine Learning plays a very deciding role in the disease prediction. Machine Learning algorithms like ensemble techniques like Random Forest, Gradient Boost, Adaboost and XGBoost,Decision tree, Logistic regression, KNN - K-nearest neighbours are used to predict the patient’s risk of getting thyroid disease. 
Web app is created to get data from users to predict the type of disease.

## How to run?

### Step-1 : Clone this repository into your local system by using below command

```bash
git clone https://github.com/guptadikshant/DetectionOfThyroid.git

```

### Step-2 Create a virtual environment using Anaconda Prompt by using below command
```bash
conda create -n <environment name> python==3.7 -y
```
```bash
conda activate <environment name>
```

### Step-3 Now install requirements.txt file by using below command
```bash
pip install -r requirements.txt
```

### Step-4 Now run the app.py file by using below command in the terminal
```bash
python app.py
```

    
## Dataset Link is below
https://www.kaggle.com/yasserhessein/thyroid-disease-data-set?select=hypothyroid.csv


## Main Page
![main_page](https://user-images.githubusercontent.com/51189309/143774452-e5525ab6-acc9-4b5d-9dc1-4d2ae4ca35ee.JPG)

## Model Predicton Page
![model prediction](https://user-images.githubusercontent.com/51189309/143774500-155d8b0a-725d-407a-ac21-30ce3cca3446.JPG)

## Prediction Saved in MongoDB Atlas
![prediction saved in mongodb](https://user-images.githubusercontent.com/51189309/143774536-f5711ed9-4ae6-4203-a1f7-d8021a4b680c.JPG)

