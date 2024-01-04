from src.entity.artifact_entity import DataIngestionArtifact
from src.logger import logging
from src.exception import FraudDetectionException
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact
import os,sys
import pandas  as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import json
from src.util.util import read_json_file
import os


from src.constant import *


class DataValidation:
    

    def __init__(self, data_validation_config:DataValidationConfig,
        data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*30}Data Valdaition log started.{'<<'*30} \n\n")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise FraudDetectionException(e,sys) from e


    def is_file_exists(self)->bool:
        try:
            logging.info("Checking if file available for validation")
            is_file_exist = False
            file_path = self.data_ingestion_artifact.data_file_path
            is_file_exist = os.path.exists(file_path)            
            is_available =  is_file_exist
            logging.info(f"Is file exists?-> {is_available}")
            
            if not is_available:
                data_file = self.data_ingestion_artifact.data_file_path
                
                message=f"File: {data_file} is not present"
                raise Exception(message)

            return is_available
        except Exception as e:
            raise FraudDetectionException(e,sys) from e

    
    def validate_dataset_schema(self)->bool:
        try:
            validation_status = False

            logging.info("Validating the provided dataset using json file")
            json_path=self.data_validation_config.schema_file_path
            json_info= read_json_file(json_path)

            NumberofColumns = json_info['NumberofColumns']
            column_dict = json_info['ColName']
            column_names=list(column_dict.keys())

            logging.info("Loading data frame from the provided dataset")
            raw_data_dir = self.data_ingestion_artifact.data_file_path
            file_name = os.listdir(raw_data_dir)[0]
            file_path = os.path.join(raw_data_dir,file_name)

            logging.info(f"Reading csv file: [{file_path}]")
            data_df = pd.read_csv(file_path)
            data_columns = data_df.columns.tolist()
            
            logging.info("checking for the number of columns")  
            if data_df.shape[1] == NumberofColumns:
                logging.info(f"Data file have column length: {NumberofColumns}")
                validation_status = True
            else:
                logging.info(f"Data file is having different column length: {NumberofColumns}")

            logging.info("checking for name of the columns")
            if data_columns == column_names:
                logging.info("Data file have same column names")
                validation_status = True
            else:
                logging.info("Data file have different column names")
                validation_status = False
                      
            return validation_status 
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
    
    def load_present_and_previous_data(self):

        try:
            raw_data_dir = self.data_ingestion_artifact.data_file_path
            file_name = os.listdir(raw_data_dir)[0]
            file_path = os.path.join(raw_data_dir,file_name)

            logging.info(f"Reading present file: [{file_path}]")
            present_data_df = pd.read_csv(file_path)

            
            logging.info("Loading data frame from the previous dataset")

            
            data_ingestion_dir = os.path.join(ROOT_DIR, 'src', 'artifact', 
                                                       DATA_INGESTION_ARTIFACT_DIR)
            
            file_names_list = os.listdir(data_ingestion_dir)
            file_names_list.remove(max(file_names_list))
            previous_file=max(file_names_list)
            
            file_dir=os.path.join(data_ingestion_dir, previous_file, 'raw_data') 

            previous_file_name = os.listdir(file_dir)[0]
            previous_file_path= os.path.join(file_dir, previous_file_name)

            logging.info(f"Reading previous file: [{previous_file_path}]")
            previous_data_df = pd.read_csv(previous_file_path)

            return present_data_df, previous_data_df

        except Exception as e:
            raise FraudDetectionException(e,sys) from e 

       

    def get_and_save_data_drift_report(self):
        try:
            profile = Profile(sections=[DataDriftProfileSection()])

            present_data_df, previous_data_df = self.load_present_and_previous_data()


            null_columns_present = present_data_df.columns[present_data_df.isnull().all()]
            null_columns_previous = previous_data_df.columns[previous_data_df.isnull().all()]
            present_data_df.drop(null_columns_present, axis=1, inplace=True)
            previous_data_df.drop(null_columns_previous, axis=1, inplace=True)
            

            profile.calculate(present_data_df, previous_data_df)

            report = json.loads(profile.json())

            report_file_path = self.data_validation_config.report_file_path
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir,exist_ok=True)

            with open(report_file_path,"w") as report_file:
                json.dump(report, report_file, indent=6)
            return report
        except Exception as e:
            raise FraudDetectionException(e,sys) from e

    def save_data_drift_report_page(self):
        try:
            dashboard = Dashboard(tabs=[DataDriftTab()])
            present_data_df, previous_data_df = self.load_present_and_previous_data()


            null_columns_present = present_data_df.columns[present_data_df.isnull().all()]
            null_columns_previous = previous_data_df.columns[previous_data_df.isnull().all()]
            present_data_df.drop(null_columns_present, axis=1, inplace=True)
            previous_data_df.drop(null_columns_previous, axis=1, inplace=True)


            dashboard.calculate(present_data_df, previous_data_df)

            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir,exist_ok=True)

            dashboard.save(report_page_file_path)
        except Exception as e:
            raise FraudDetectionException(e,sys) from e

    def is_data_drift_found(self)->bool:
        try:
            report = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            return True
        except Exception as e:
            raise FraudDetectionException(e,sys) from e

            

    def initiate_data_validation(self)->DataValidationArtifact :
        try:
            self.is_file_exists()
            self.validate_dataset_schema()
            self.is_data_drift_found()

            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.data_validation_config.schema_file_path,
                report_file_path=self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_path,
                is_validated=True,
                message="Data Validation performed successully."
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise FraudDetectionException(e,sys) from e


    def __del__(self):
        logging.info(f"{'>>'*30}Data Valdaition log completed.{'<<'*30} \n\n")
        