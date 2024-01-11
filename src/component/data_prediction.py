import sys, os
import tarfile
import numpy as np
import pandas as pd
from six.moves import urllib
from src.logger import logging
from src.constant import *
from src.exception import FraudDetectionException
from src.entity.artifact_entity import DataPredictionArtifact, ModelPusherArtifact
from src.util.util import  read_json_file, save_data, scale_numerical_columns, \
    load_object, find_correct_model_file
from sklearn.impute import SimpleImputer
from src.component.data_transformation import CustomEncoder
from src.entity.config_entity import PredictionConfig, DataTransformationConfig, \
    DataValidationConfig, ModelPusherConfig

ROOT_DIR = os.getcwd()
PIPELINE_FOLDER_NAME = "src"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)



class DataPrediction:

    def __init__(self,data_prediction_config:PredictionConfig, data_validation_config:DataValidationConfig,
                 data_transformation_config: DataTransformationConfig, model_pusher_config: ModelPusherConfig):
        try:
            logging.info(f"{'>>'*20}Bulk Data prediction log started.{'<<'*20} ")
            self.data_prediction_config = data_prediction_config
            self.data_validation_config=data_validation_config
            self.data_transformation_config=data_transformation_config
            self.model_pusher_config=model_pusher_config

        except Exception as e:
            raise FraudDetectionException(e,sys)
    
    def download_and_extract_data(self) -> str:
        try:
            #extraction remote url to download dataset
            download_url = self.data_prediction_config.dataset_download_url
            #folder location to download file
            tgz_download_dir = self.data_prediction_config.tgz_download_dir
            os.makedirs(tgz_download_dir,exist_ok=True)
            file_name = os.path.basename(download_url)
            tgz_file_path = os.path.join(tgz_download_dir, file_name)
            logging.info(f"Downloading file from :[{download_url}] into :[{tgz_file_path}]")
            urllib.request.urlretrieve(download_url, tgz_file_path)
            logging.info(f"File :[{tgz_file_path}] has been downloaded successfully.")

            raw_data_dir = self.data_prediction_config.raw_data_dir
            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)
            os.makedirs(raw_data_dir,exist_ok=True)
            logging.info(f"Extracting tgz file: [{tgz_file_path}] into dir: [{raw_data_dir}]")
            with tarfile.open(tgz_file_path) as tgz_file_obj:
                tgz_file_obj.extractall(path=raw_data_dir)
            logging.info(f"Extraction completed")

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
            raw_data_dir = self.data_prediction_config.raw_data_dir
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

    def data_transformation(self,data):
        try:

            data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            null_columns = data.columns[data.isnull().all()]
            data.drop(null_columns, axis=1, inplace=True)
            data['fraud_reported'] = None 

            json_path=self.data_validation_config.schema_file_path
            json_info= read_json_file(json_path)
            columns=json_info[COLUMNS_TO_REMOVE]
         
            new_df=data.drop(labels=columns, axis=1) 
            # drop the labels specified in the columns
            new_df.replace('?',np.NaN,inplace=True)
            numerical_columns = [x for x in json_info[NUMERICAL_COLUMN_KEY] 
                                 if x not in json_info[COLUMNS_TO_REMOVE] ]
            categorical_columns = [x for x in json_info[CATEGORICAL_COLUMN_KEY] 
                                   if x not in json_info[COLUMNS_TO_REMOVE] ]
            
            logging.info('Imputing numerical values')
            num_imputer = SimpleImputer(strategy='mean')
            new_df[numerical_columns] = num_imputer.fit_transform(new_df[numerical_columns])    
            logging.info('Imputing categorical values')
            cat_imputer = SimpleImputer(strategy='most_frequent')
            new_df[categorical_columns] = cat_imputer.fit_transform(new_df[categorical_columns])
            custom_encoding_obj=CustomEncoder()
            logging.info('Cusom encoding the data')
            new_df=custom_encoding_obj.fit_transform(new_df)
            ohe_columns=json_info[COLUMNS_FOR_OHE]
            logging.info('One Hot Eencoding for remaining categorical data')
            encoded_data = pd.get_dummies(new_df, columns=ohe_columns, drop_first=True)

            return encoded_data.astype(float) 
        except Exception as e:
            raise FraudDetectionException(e,sys) from e 


    def data_predict(self, data):
        try:
            models_root_dir=self.model_pusher_config.export_dir_path
            parent_dir=os.path.dirname(models_root_dir)


            logging.info(f"model dir: {parent_dir}")

            folder_name = list(map(int, os.listdir(parent_dir)))
            latest_model_dir = os.path.join(parent_dir, f"{max(folder_name)}")
            
            #self.model_pusher_config.export_dir_path
            cluster_object_file_path =os.path.join(latest_model_dir, 'KMeans')

            file_name = os.listdir(cluster_object_file_path)[0]
            cluster_model = os.path.join(cluster_object_file_path,file_name)
            kmeans=load_object(cluster_model)
            df=data.drop('fraud_reported', axis=1)
            clusters=kmeans.predict(df)
            df['clusters']=clusters
            no_of_clusters=df['clusters'].unique()
            predictions=[]
            for i in no_of_clusters:
                cluster_data= df[df['clusters']==i]
                cluster_data = cluster_data.drop(['clusters'],axis=1)
                model_name = find_correct_model_file(path=latest_model_dir, cluster_number=int(i))
                model_dir = os.path.join(latest_model_dir,model_name)
                file_name = os.listdir(model_dir)[0]
                
                model = load_object(os.path.join(model_dir, file_name))
                array = scale_numerical_columns(cluster_data)
                result=(model.predict(array))
                for res in result:
                    if res==0:
                        predictions.append('N')
                    else:
                        predictions.append('Y')

            final= pd.DataFrame(list(zip(predictions)),columns=['Fraud_Predictions'])

            return final

        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        

    def initiate_data_prediction(self)-> DataPredictionArtifact:
        try:

            self.download_and_extract_data()
            validation_status=self.validate_dataset_schema

            raw_data_dir = self.data_prediction_config.raw_data_dir
            file_name = os.listdir(raw_data_dir)[0]
            file_path = os.path.join(raw_data_dir,file_name)
            logging.info(f"Reading csv file: [{file_path}]")
            data_df = pd.read_csv(file_path)

            transformed_data=self.data_transformation(data_df)

            predict_df=self.data_predict(transformed_data)

            final_df=pd.concat([data_df, predict_df], axis=1)
            predict_path=self.data_prediction_config.predicted_path

            if os.path.exists(predict_path):
                os.remove(predict_path)
            os.makedirs(predict_path,exist_ok=True)

            final_df.to_csv(os.path.join(predict_path, "Final_Result.csv" ))

            data_prediction_artifact = DataPredictionArtifact(is_predicted= True, export_data_file_path=predict_path)
            logging.info(f"Data Prediction artifact:[{data_prediction_artifact}]")

            return data_prediction_artifact         
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
    
    def __del__(self):
        logging.info(f"{'>>'*20}Data prediction log completed.{'<<'*20} \n\n")