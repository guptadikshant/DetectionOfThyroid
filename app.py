from TrainingModels import training

def train_route():
    train_obj = training.TrainModel()
    print("object created")
    train_obj.separate_train_test_split()
    print("separate_train_test_split successfully created")
    train_obj.preprocess_data()
    print("preprocess_data successfully created")
    train_obj.best_model_select()
    print("best_model_select successfully created")

if __name__ == '__main__':
    train_route()