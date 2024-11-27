import os
import pandas as pd
from shutil import rmtree
from dataclasses import dataclass
from src.logger import logging
from src.exception_handler import DataIngestionException
from sklearn.model_selection import train_test_split


# Add all the configuration and inputs for Data Ingestion component
# inside a dataclass
@dataclass
class DataIngestionConfig:
    test_size: float = 0.2
    random_state: int = 45
    artifact_folder = os.path.join(os.getcwd(), "artifacts")
    raw_data_file_path: str = os.path.join(artifact_folder, "raw_data.csv")
    train_data_file_path: str = os.path.join(artifact_folder, "train_data.csv")
    test_data_file_path: str = os.path.join(artifact_folder, "test_data.csv")
    data_source_path: str = os.path.join("notebook", "data", "stud.csv")


class DataIngestion:
    def __init__(self):
        self.__create_necessary_directories()

    def initiate_data_ingestion(self):
        logging.info("Initiated the DataIngestion")
        try:
            df = pd.read_csv(DataIngestionConfig.data_source_path)
            logging.info("Data successfully read into data frame")

            train_df, test_df = train_test_split(
                df,
                test_size=DataIngestionConfig.test_size,
                random_state=DataIngestionConfig.random_state,
            )

            logging.info("Data successfully split into Train and Test")
            df.to_csv(DataIngestionConfig.raw_data_file_path, index=False, header=True)
            train_df.to_csv(
                DataIngestionConfig.train_data_file_path, index=False, header=True
            )
            test_df.to_csv(
                DataIngestionConfig.test_data_file_path, index=False, header=True
            )
            logging.info("All Data successfully saved to all artifacts file")

            return (
                DataIngestionConfig.train_data_file_path,
                DataIngestionConfig.test_data_file_path,
            )

        except Exception as error:
            raise DataIngestionException(error)

    def __create_necessary_directories(self):
        try:
            if os.path.exists(DataIngestionConfig.artifact_folder):
                logging.debug(f"{DataIngestionConfig.artifact_folder} Deleted")
                rmtree(DataIngestionConfig.artifact_folder)

            os.makedirs(DataIngestionConfig.artifact_folder)
            logging.info("Folder creation successful")

        except Exception as error:
            raise OSError(
                "Error encountered while creating necessary directories for Data Ingestion,"
                "ERROR: {}".format(error)
            )


if __name__ == "__main__":
    obj = DataIngestion()
    print(obj.initiate_data_ingestion())
