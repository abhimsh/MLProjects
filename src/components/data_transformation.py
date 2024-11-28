import os
from dataclasses import dataclass

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception_handler import DataTransformationException
from src.components.data_ingestion import DataIngestion
from src.utils import save_object
import pandas as pd
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join(
        os.getcwd(), 
        "artifacts",
        "preprocessor.pkl"
    )
    numerical_columns = [
        "reading_score", 
        "writing_score"
        ]
    categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
    target_column_name = "math_score"

class DataTransformation:
    
    def __create_preprocessor(self) -> None:
        """
        create the preprocessor containing transformation of all columns in the DataSet 
        """
        try:

            logging.debug(f"Numerical columns: {DataTransformationConfig.numerical_columns}")
            logging.debug(f"Categorical columns: {DataTransformationConfig.categorical_columns}")

            num_pipeline = Pipeline(
                steps=[
                    ("Impute", SimpleImputer(strategy="median")),
                    ("Standardization", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                        ("Impute", SimpleImputer(strategy="most_frequent")),
                        ("Encoding", OneHotEncoder())
                ]
            )

            logging.info("Pipelines created successfully")

            preprocessor = ColumnTransformer(
                [
                    ("Cat_transformation", cat_pipeline, DataTransformationConfig.categorical_columns),
                    ("Numerical_transformation", num_pipeline, DataTransformationConfig.numerical_columns)
                ]
            )
            
            logging.info("preprocessor created successfully")
            
            return preprocessor
        
        except Exception as error:
            logging.exception(f"DataTransformationException: {error}")
            raise DataTransformationException(f"Exception Encocountered while creating preprocessor, Error: {error}")

    def initiate_data_transformation(self, train_file_path:str, test_file_path: str):
        try:
            # Read training and testing data
            train_data = pd.read_csv(train_file_path)
            test_data = pd.read_csv(test_file_path)
            logging.info("The taringing and test data read successfully from path")

            # Get the preprocessor obj
            preproceesor_obj = self.__create_preprocessor()
            logging.info("Preprocessor object created successfully")

            # Split the target column and independent column from train and test data for 
            # applying preprocessing
            train_x_data = train_data.drop(columns=[DataTransformationConfig.target_column_name], axis=1)
            train_y_data = train_data[DataTransformationConfig.target_column_name]

            test_x_data = test_data.drop(columns=[DataTransformationConfig.target_column_name], axis=1)
            test_y_data = test_data[DataTransformationConfig.target_column_name]

            # apply transformation to the data
            transformed_x_train_data = preproceesor_obj.fit_transform(train_x_data)
            transformed_x_test_data = preproceesor_obj.transform(test_x_data)

            # Combine the independent and dependent features into single array
            train_arr = np.hstack((transformed_x_train_data, train_y_data.values.reshape(-1, 1)))
            test_arr = np.hstack((transformed_x_test_data, test_y_data.values.reshape(-1, 1)))

            save_object(file_path=DataTransformationConfig.preprocessor_file_path,
                        obj=preproceesor_obj)

            return train_arr, test_arr, DataTransformationConfig.preprocessor_file_path
        except Exception as error:
            logging.exception(f"Exception encountered while Data transformation, Error: {error}")
            raise DataTransformationException(error)
        

if __name__ == "__main__":

    ingestion_obj = DataIngestion()
    train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()

    obj = DataTransformation()
    train_arr, test_arr, preprocessor_obj = obj.initiate_data_transformation(
        train_data_path, 
        test_data_path
        )