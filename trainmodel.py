from TrainingModels import training

def train_route():
    train_obj = training.TrainModel()
    train_obj.separate_train_test_split()
    train_obj.preprocess_data()
    train_obj.best_model_select()


if __name__ == '__main__':
    train_route()