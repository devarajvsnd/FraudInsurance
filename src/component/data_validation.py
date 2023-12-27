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
        
    '''
    def load_previous_data(self):

        try:

            logging.info("Loading data frame from the previous dataset")

            previous_data_dir= self.data_ingestion_artifact.data_file_path

        





            pass

        except Exception as e:
            raise FraudDetectionException(e,sys) from e 



        
        

    def get_and_save_data_drift_report(self):
        try:
            profile = Profile(sections=[DataDriftProfileSection()])

            train_df,test_df = self.get_train_and_test_df()

            profile.calculate(train_df,test_df)

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
            train_df,test_df = self.get_train_and_test_df()
            dashboard.calculate(train_df,test_df)

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

            '''

    def initiate_data_validation(self)->DataValidationArtifact :
        try:
            self.is_file_exists()
            self.validate_dataset_schema()
          # self.is_data_drift_found()

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
        