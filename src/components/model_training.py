import os
from dataclasses import dataclass
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.utils import save_object, evaluate_models
from src.exception_handler import ModelTrainingException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.logger import logging

@dataclass
class ModelTrainingConfig:
    best_model_file_path = os.path.join("artifacts", "model.pkl") 
    all_models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge Regression": Ridge(),
        "Elastic Net": ElasticNet(),
        "Decision Tree": DecisionTreeRegressor(),
        "K-Nearest Neighbour": KNeighborsRegressor(),
        "Random Forest": RandomForestRegressor(),
        "AdaBoost": AdaBoostRegressor(),
        "GradientBoost": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor(),
        "CatBoost": CatBoostRegressor()
    }

    model_params = {
            "Decision Tree": {
                'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                'splitter':['best','random'],
                'max_features':['sqrt','log2', None],
            },
            "Random Forest":{
                'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                'max_features':['sqrt','log2', 1.0],
                'n_estimators': [8, 16, 32, 64, 100, 128, 256]
            },
            "GradientBoost":{
                'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                'learning_rate':[.1, .01, .05, .001],
                'subsample':[0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0],
                'criterion':['squared_error', 'friedman_mse'],
                'max_features':['sqrt','log2', None],
                'n_estimators': [8, 16, 32, 64, 100, 128, 256]
            },
            "XGBoost":{
                'learning_rate':[.1,.01,.05,.001],
                'n_estimators': [8,16,32,64,128,256]
            },
            "CatBoost":{
                'depth': [6,8,10, None],
                'learning_rate': [0.01, 0.05, 0.1, None],
                'iterations': [30, 50, 100, None]
            },
            "AdaBoost":{
                'learning_rate':[1.0, .1, .01, 0.5, .001],
                'loss':['linear','square','exponential'],
                'n_estimators': [8, 16, 32, 64, 128, 256, None]
            },
            "K-Nearest Neighbour":
            {
                "n_neighbors": (1, 3, 4, 5, 6, 7),
                "weights": ("uniform", "distance"),
                "algorithm": ("ball_tree", "kd_tree", "brute", "auto"),
                "p":(1,2)
            },
            "Linear Regression":{},
            "Lasso": {},
            "Ridge Regression":  {},
            "Elastic Net":  {}
            }

    performance_threshold = 0.6


class ModelTrainer:
    def train_the_model(self, X_train_array:NDArray, X_test_array:NDArray):
        try:
            
            # Split the train and test array into independent and dependent feature
            # last column is the target
            X_train, X_test, y_train, y_test = (
                X_train_array[:, :-1],
                X_test_array[:, :-1],
                X_train_array[:, 1],
                X_test_array[:, 1],
            )
            
            logging.info("Data successfully split into independent and target feature")

            # initiate model training
            all_models_performance: dict = evaluate_models(X_train,
                                                           y_train,
                                                           X_test,
                                                           y_test,
                                                           models=ModelTrainingConfig.all_models,
                                                           model_parameters=ModelTrainingConfig.model_params,
                                                           score="r2_score")
            
            logging.info("All the models have been trained successfully")

            # get the model with best score
            best_model = sorted(all_models_performance.items(), 
                                key=lambda item: item[1][-1])[-1]
            
            logging.info(f"best model: {best_model}")
            
            # if the best model's score is less than threshold, then reject 
            if best_model[-1][-1] < ModelTrainingConfig.performance_threshold:
                raise ModelTrainingException("No Good Model found!!!!")


            # Save the model to file for further use
            save_object(file_path=ModelTrainingConfig.best_model_file_path,
                        obj=ModelTrainingConfig.all_models[best_model[0]])

        except Exception as error:
            logging.exception(f"Exception encounterd while Performing Model training, Error:{error}")
            raise ModelTrainingException(f"Exception encounterd while Performing Model training, Error:{error}")


if __name__ == "__main__":

    ingestion_obj = DataIngestion()
    train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()

    transformation_obj = DataTransformation()
    train_arr, test_arr, preprocessor_obj = \
    transformation_obj.initiate_data_transformation(
        train_data_path, 
        test_data_path
        )
    
    model_training_obj = ModelTrainer()
    model_training_obj.train_the_model(train_arr, test_arr)