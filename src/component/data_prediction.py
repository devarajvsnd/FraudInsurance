from src.entity.config_entity import PredictionConfig, DataTransformationConfig, \
    DataValidationConfig, ModelTrainerConfig, ModelPusherConfig
import sys,os
from src.exception import FraudDetectionException
from src.logger import logging
from src.entity.artifact_entity import DataPredictionArtifact
import tarfile
import numpy as np
from six.moves import urllib
import pandas as pd
from src.util.util import  read_json_file, save_data, save_object, load_data, save_model, \
    scale_numerical_columns, load_object, find_correct_model_file
from sklearn.impute import SimpleImputer
from src.component.data_transformation import CustomEncoder



from src.constant import *

class DataPrediction:

    def __init__(self,data_prediction_config:PredictionConfig, data_validation_config:DataValidationConfig,
                 data_transformation_config: DataTransformationConfig, model_pusher_config: ModelPusherConfig ):
        try:
            logging.info(f"{'>>'*20}Data prediction log started.{'<<'*20} ")
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

            json_path=self.data_validation_config.schema_file_path
            json_info= read_json_file(json_path)
            columns=json_info[COLUMNS_TO_REMOVE]
         
            useful_data=data.drop(labels=columns, axis=1) 
            # drop the labels specified in the columns
            useful_data.replace('?',np.NaN,inplace=True)
            numerical_columns = [x for x in json_info[NUMERICAL_COLUMN_KEY] 
                                 if x not in json_info[COLUMNS_TO_REMOVE] ]
            categorical_columns = [x for x in json_info[CATEGORICAL_COLUMN_KEY] 
                                   if x not in json_info[COLUMNS_TO_REMOVE] ]
            
            logging.info('Imputing numerical values')
            num_imputer = SimpleImputer(strategy='mean')
            data[numerical_columns] = num_imputer.fit_transform(data[numerical_columns])    
            logging.info('Imputing categorical values')
            cat_imputer = SimpleImputer(strategy='most_frequent')
            data[categorical_columns] = cat_imputer.fit_transform(data[categorical_columns])
            custom_encoding_obj=CustomEncoder()
            logging.info('Cusom encoding the data')
            data=custom_encoding_obj.fit_transform(data)
            ohe_columns=json_info[COLUMNS_FOR_OHE]
            logging.info('One Hot Eencoding for remaining categorical data')
            encoded_data = pd.get_dummies(data, columns=ohe_columns, drop_first=True)

            data=encoded_data.astype(float) 
            array = scale_numerical_columns(encoded_data)
            return array
        except Exception as e:
            raise FraudDetectionException(e,sys) from e 


    def data_predict(self, data):
        try:
            models_file_location=self.model_pusher_config.export_dir_path
            cluster_object_file_path =os.path.join(models_file_location, 'KMeans')

            file_name = os.listdir(cluster_object_file_path)[0]
            cluster_model = os.path.join(cluster_object_file_path,file_name)
            kmeans=load_object(cluster_model)
            clusters=kmeans.predict(data)
            data['clusters']=clusters
            clusters=data['clusters'].unique()
            predictions=[]
            for i in clusters:
                cluster_data= data[data['clusters']==i]
                cluster_data = cluster_data.drop(['clusters'],axis=1)
                model_name = find_correct_model_file(path=models_file_location, cluster_number=int(i))
                model_dir = os.path.join(models_file_location,model_name)
                file_name = os.listdir(model_dir)[0]
                
                model = load_object(os.path.jpin(model_dir, file_name))

                result=(model.predict(cluster_data))
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

            transformed_data_array=self.data_transformation(data_df)

            predict_df=self.data_predict(transformed_data_array)

            final_df=pd.concat([data_df, predict_df])
            predict_path=self.data_prediction_config.predicted_path
            save_data(predict_path, final_df)


            data_prediction_artifact = DataPredictionArtifact(is_predicted= True, export_data_file_path=predict_path)
            logging.info(f"Data Prediction artifact:[{data_prediction_artifact}]")

            return data_prediction_artifact         
        
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
    


    def __del__(self):
        logging.info(f"{'>>'*20}Data prediction log completed.{'<<'*20} \n\n")