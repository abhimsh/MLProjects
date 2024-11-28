import pandas as pd

from src.utils import load_object
from src.components.data_transformation import DataTransformationConfig
from src.components.model_training import ModelTrainingConfig
from src.exception_handler import PredictPipelineException
from src.logger import logging


class PredictionDataClass:

    def get_test_data_as_data_frame(self,
                                    gender: str, 
                                    race_ethnicity: str, 
                                    parental_level_of_education: str,
                                    lunch: str,
                                    test_preparation_course: str,
                                    reading_score: int,
                                    writing_score: int):
        data_frame = pd.DataFrame(
            {
               "gender": [gender],
               "race_ethnicity": [race_ethnicity],
               "parental_level_of_education": [parental_level_of_education],  
               "lunch": [lunch],  
               "test_preparation_course": [test_preparation_course],  
               "reading_score": [reading_score],  
               "writing_score": [writing_score],  
            }
        )
        return data_frame

    def predict(self,
                gender: str, 
                race_ethnicity: str, 
                parental_level_of_education: str,
                lunch: str,
                test_preparation_course: str,
                reading_score: int,
                writing_score: int):
        try:
            model = load_object(ModelTrainingConfig.best_model_file_path)
            preprocessor = load_object(DataTransformationConfig.preprocessor_file_path)

            new_data = self.get_test_data_as_data_frame(
                                        gender, 
                                        race_ethnicity, 
                                        parental_level_of_education,
                                        lunch,
                                        test_preparation_course,
                                        reading_score,
                                        writing_score
                                        )
            logging.debug(new_data)
            processed_data = preprocessor.transform(new_data)
            logging.debug(f"Transformed data: {processed_data}")
            predicted_math_score = model.predict(processed_data)
            logging.debug(f"Predicted  data: {predicted_math_score}")
            
            return predicted_math_score
        
        except Exception as error:
            logging.exception("Error Encounterd while writing obj to file")
            raise PredictPipelineException(error)