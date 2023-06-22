import os
import sys
import shutil
from src.logger import logging
from src.exception import CustomException
from src.components.Data_Processing import DataProcessing
# from src.components.Model_training import ModelTrainer
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path_class1: str = "artifacts/data/healthy_data/"
    raw_data_path_class2: str = "artifacts/data/unhealthy_data/"


class DataIngestion:
    def __init__(self):
        # Instantiating an instance of DataIngestionConfig
        self.ingestion_config = DataIngestionConfig()

    def initialize_data_ingestion(self):
        logging.info("Starting the data ingestion")

        try:
            # Making destination directories
            os.makedirs(self.ingestion_config.raw_data_path_class1, exist_ok=True)
            os.makedirs(self.ingestion_config.raw_data_path_class2, exist_ok=True)

            # Reading images from local directory, but here this resource could be anything
            source_dir_path = os.path.join('notebook','data','')
            
            # Creating the list of subdirectories in teh main directory
            sub_dirs = [subdir for subdir in os.listdir(source_dir_path) if os.path.isdir(os.path.join(source_dir_path, subdir))]

            # Let's now iterate through each directory
            for sub_dir in sub_dirs:
                if sub_dir == "Apple___healthy":
                    files_list = os.listdir(os.path.join(source_dir_path, sub_dir))
                    for file in files_list:
                        source_path = os.path.join(source_dir_path, sub_dir, file)
                        destination_path = os.path.join(self.ingestion_config.raw_data_path_class1, file)

                        # Copying images to destination directory
                        shutil.copyfile(source_path, destination_path)
                else:
                    files_list = os.listdir(os.path.join(source_dir_path, sub_dir))
                    for file in files_list:
                        source_path = os.path.join(source_dir_path, sub_dir, file)
                        destination_path = os.path.join(self.ingestion_config.raw_data_path_class2, file)

                        # Copying images to destination directory
                        shutil.copyfile(source_path, destination_path)

            logging.info("Loaded images into the destination directory")
            return (
                self.ingestion_config.raw_data_path_class1,
                self.ingestion_config.raw_data_path_class2
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    healthy_imgs, unhealthy_imgs = data_ingestion.initialize_data_ingestion()

    # Now using the healthy_imgs and unhealthy_imgs path we will load the images in the tensorflow dataset and apply preprocessing
    # data_transformation = DataProcessing()
    # print(data_transformation.initialize_process_train('artifacts'))

    # modeltrainer = ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_ds, val_ds))

    
