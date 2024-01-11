import yaml
from src.exception import FraudDetectionException
import os,sys
import numpy as np
import dill
import pandas as pd
from src.constant import *
import json, pickle, shutil
from sklearn.preprocessing import StandardScaler


def write_yaml_file(file_path:str,data:dict=None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,"w") as yaml_file:
            if data is not None:
                yaml.dump(data,yaml_file)
    except Exception as e:
        raise FraudDetectionException(e,sys)


def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise FraudDetectionException(e,sys) from e


def read_json_file(file_path:str)->dict:
    """
    Reads a json file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    except Exception as e:
        raise FraudDetectionException(e,sys) from e



def save_data(file_path: str, dataframe: pd.DataFrame):

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dataframe.to_csv(file_path, index=False)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e


def save_image(file_path: str, dataframe: pd.DataFrame):
    """
    Save numpy data to file
    file_path: str location of file to save
        """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dataframe.to_csv(file_path, index=False)
    except Exception as e:
        raise FraudDetectionException(e, sys) from e




def save_model(file_path:str, model, filename:str):


    try:
        #dir_path = os.path.dirname(file_path)
        #os.makedirs(dir_path, exist_ok=True)

        path = os.path.join(file_path, filename) #create seperate directory for each cluster
        if os.path.isdir(path): #remove previously existing models for each clusters
            shutil.rmtree(file_path)
            os.makedirs(path)
        else:
            os.makedirs(path) #
        with open(path +'/' + filename + '.sav', 'wb') as f:
            pickle.dump(model, f) # save the model to file'''

    except Exception as e:
        raise FraudDetectionException(e, sys) from e




def save_object(file_path:str,obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise FraudDetectionException(e,sys) from e


def load_object(file_path:str):
    """
    file_path: str
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise FraudDetectionException(e,sys) from e


def load_data(path: str, schema_file_path: str) -> pd.DataFrame:
    try:
        datatset_schema = read_json_file(schema_file_path)

        schema = datatset_schema[DATASET_SCHEMA_COLUMNS_KEY]

        file_name = os.listdir(path)[0]
        file_path = os.path.join(path,file_name)

        dataframe = pd.read_csv(file_path)

        error_messgae = ""


        for column in dataframe.columns:
            if column in list(schema.keys()):
                dataframe[column].astype(schema[column])
            else:
                error_messgae = f"{error_messgae} \nColumn: [{column}] is not in the schema."
        if len(error_messgae) > 0:
            raise Exception(error_messgae)
        return dataframe

    except Exception as e:
        raise FraudDetectionException(e,sys) from e
    



def scale_numerical_columns(data):


    '''num_df = data[['months_as_customer', 'policy_deductable', 'umbrella_limit',
                      'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
                      'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim',
                      'property_claim',
                      'vehicle_claim']]'''

    scaler = StandardScaler()
    #scaled_data = scaler.fit_transform(num_df)

    scaled_data = scaler.fit_transform(data)

    scaled_df = pd.DataFrame(data=scaled_data, columns=data.columns,index=data.index)
    #data.drop(columns=scaled_df.columns, inplace=True)
    #data = pd.concat([scaled_df, data], axis=1)

    return scaled_df




def find_correct_model_file(cluster_number, path):

    list_of_files = os.listdir(path)
    for file in list_of_files:
        try:
            if (file.index(str(cluster_number))!=-1):
                model_name=file
        except:
            continue
    model_name=model_name.split('.')[0]
    return model_name 


