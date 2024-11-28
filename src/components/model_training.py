import os
from dataclasses import dataclass
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

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
        "XGBoost": XGBRegressor()
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
                                                           score="r2_score")
            
            logging.info("All the models have been trained successfully")

            # get the model with best score
            best_model = sorted(all_models_performance.items(), 
                                key=lambda item: item[1])[-1]
            
            logging.info(f"best model: {best_model}")
            
            # if the best model's score is less than threshold, then reject 
            if best_model[-1] < ModelTrainingConfig.performance_threshold:
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