import dill
import os
from numpy.typing import NDArray
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception_handler import UtilsException



def save_object(file_path: str, obj) -> None:
    """
    Will save the python object into file using the pickling
    """
    try:
        logging.debug(f"({file_path}, {repr(obj)})")
        # Create the folder if not present
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the object to file
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.debug(f"Successfully written {repr(file_obj)} into {file_path}")

    except Exception as error:
        logging.exception("Error Encounterd while writing obj to file")
        raise UtilsException(error)
    

def evaluate_models(X_train: NDArray, 
                    y_train: NDArray, 
                    X_test: NDArray,
                    y_test: NDArray, 
                    models: dict,
                    model_parameters: dict,
                    score: str) -> dict:
    try:
        scoring_dict = {"r2_score": r2_score, "mae": mean_absolute_error, 
                        "mse": mean_squared_error}
        
        # Check if the valid scoring parameter has been received
        assert scoring_dict.get(score, None) is not None, "Invalid scoring provided"
        
        model_performance = {}
        logging.debug(f"performance matrix selected: {scoring_dict[score]}")
        
        # Check the performance of all received models against score
        for model_name, model_obj in models.items():
            
            logging.info(f"Start Model training for {model_name}")
            cv = GridSearchCV(model_obj, 
                              param_grid=model_parameters[model_name], 
                              cv=3,
                              n_jobs=-1,
                              scoring="accuracy",
                              verbose=3)
            
            cv.fit(X_train, y_train)

            # Retrain the models with best parameters found during Hyper parameter tuning
            model_obj = model_obj.set_params(**cv.best_params_)
            model_obj.fit(X_train, y_train)
            y_predict = model_obj.predict(X_test)

            # Calculate metrics
            performance = scoring_dict[score](y_test, y_predict)
            model_performance[model_name] = (model_obj, performance)
            logging.debug(model_performance)
            logging.debug(f"{model_name} - {performance} - {cv.best_params_}")

        logging.info("All model tarining completed!!!")
        logging.debug(model_performance)
        return model_performance

    except Exception as error:
        logging.exception(f"Exception encounterd while training the models: {error}")
        raise UtilsException(f"ModelTrainingError: {error}")


def load_object(file_path: str):
    """
    Will load the python object from file using the pickling
    """
    try:
        logging.debug(f"({file_path})")
        
        # Create the folder if not present
        if not os.path.exists(file_path):
            raise FileNotFoundError("Invaid Path provided!!, {file_path}")
        
        # Write the object to file
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.debug(f"Successfully Loaded {repr(file_obj)} from {file_path}")

        return obj
    
    except Exception as error:
        logging.exception(f"Error Encounterd while Reading obj to file, Error: {error}")
        raise UtilsException(error)
    