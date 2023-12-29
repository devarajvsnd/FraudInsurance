from cgi import test
from sklearn import preprocessing
from src.exception import FraudDetectionException
from src.logger import logging
from src.entity.config_entity import DataPreprocessingConfig 
from src.entity.artifact_entity import DataIngestionArtifact, DataPreprocessingArtifact,\
      DataValidationArtifact, DataTransformationArtifact
import sys,os
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
from src.constant import *
from src.util.util import read_json_file,save_object,save_numpy_array_data,load_data

from sklearn_pandas import CategoricalImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from os import listdir


class CustomEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_mapped = X.copy()
        
        # Mapping for 'policy_csl'
        policy_csl_mapping = {'100/300': 1, '250/500': 2.5, '500/1000': 5}
        X_mapped['policy_csl'] = X_mapped['policy_csl'].map(policy_csl_mapping)
        
        # Mapping for 'insured_education_level'
        education_level_mapping = {
            'JD': 1, 'High School': 2, 'College': 3,
            'Masters': 4, 'Associate': 5, 'MD': 6, 'PhD': 7
        }
        X_mapped['insured_education_level'] = X_mapped['insured_education_level'].map(education_level_mapping)
        
        # Mapping for 'incident_severity'
        incident_severity_mapping = {
            'Trivial Damage': 1, 'Minor Damage': 2,
            'Major Damage': 3, 'Total Loss': 4
        }
        X_mapped['incident_severity'] = X_mapped['incident_severity'].map(incident_severity_mapping)
        
        # Mapping for 'insured_sex'
        sex_mapping = {'FEMALE': 0, 'MALE': 1}
        X_mapped['insured_sex'] = X_mapped['insured_sex'].map(sex_mapping)
        
        # Mapping for 'property_damage'
        property_damage_mapping = {'NO': 0, 'YES': 1}
        X_mapped['property_damage'] = X_mapped['property_damage'].map(property_damage_mapping)
        
        # Mapping for 'police_report_available'
        police_report_mapping = {'NO': 0, 'YES': 1}
        X_mapped['police_report_available'] = X_mapped['police_report_available'].map(police_report_mapping)

        return X_mapped














class DataPreprocessing:

    def __init__(self, data_preprocessing_config: DataPreprocessingConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            logging.info(f"{'>>' * 30}Data Preprocessing log started.{'<<' * 30} ")
            self.data_preprocessing_config= data_preprocessing_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        


    def remove_unwanted_spaces(self)-> pd.DataFrame:
        """
        This method removes the unwanted spaces from a pandas dataframe.
        Output: A pandas DataFrame after removing the spaces.
        On Failure: Raise Exception"""
        try:

            logging.info("Loading data frame from the provided dataset")
            raw_data_dir = self.data_ingestion_artifact.data_file_path
            file_name = os.listdir(raw_data_dir)[0]
            file_path = os.path.join(raw_data_dir,file_name)

            logging.info(f"Reading csv file: [{file_path}]")
            data=pd.read_csv(file_path)

            logging.info("Entered the remove_unwanted_spaces method of the Preprocessor class")
            data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

            return data
        
        except Exception as e:
            raise FraudDetectionException(e,sys) from e


    def remove_columns(self, data)-> pd.DataFrame:
        """
        Method Name: remove_columns
        Description: This method removes the given columns from a pandas dataframe.
        Output: A pandas DataFrame after removing the specified columns.
        On Failure: Raise Exception"""
        
        try:
            logging.info('Entered the remove_columns method of the Preprocessor class')

            # Dropping the columns with all the null values
            null_columns = data.columns[data.isnull().all()]
            data.drop(null_columns, axis=1, inplace=True)

            json_path=self.data_validation_artifact.schema_file_path
            json_info= read_json_file(json_path)
            columns=json_info['columns_to_remove']
         
            useful_data=data.drop(labels=columns, axis=1) 
            # drop the labels specified in the columns

            data.replace('?',np.NaN,inplace=True)
           
            return useful_data
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        

     

    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_json_file(file_path=schema_file_path)

            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]


            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler())
            ]
            )

            cat_pipeline = Pipeline(steps=[
                 ('impute', SimpleImputer(strategy="most_frequent")),
                 ('encoder_generator', CustomEncoder()),
                 ('scaler', StandardScaler(with_mean=False))
            ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessing = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns),
            ])
            return preprocessing

        except Exception as e:
            raise FraudDetectionException(e,sys) from e   




    '''

    def encode_categorical_columns(self,data):
        """
        Method Name: encode_categorical_columns
        Description: This method encodes the categorical values to numeric values.
        Output: dataframe with categorical values converted to numerical values
        On Failure: Raise Exception"""
            
        try:    
            logging.info('Entered the encode_categorical_columns method of the Preprocessor class')
            self.data=data
        
            cat_df = data.select_dtypes(include=['object']).copy()
            cat_df['policy_csl'] = cat_df['policy_csl'].map({'100/300': 1, '250/500': 2.5, '500/1000': 5})
            cat_df['insured_education_level'] = cat_df['insured_education_level'].map(
                {'JD': 1, 'High School': 2, 'College': 3, 'Masters': 4, 'Associate': 5, 'MD': 6, 'PhD': 7})
            cat_df['incident_severity'] = cat_df['incident_severity'].map(
                {'Trivial Damage': 1, 'Minor Damage': 2, 'Major Damage': 3, 'Total Loss': 4})
            cat_df['insured_sex'] = cat_df['insured_sex'].map({'FEMALE': 0, 'MALE': 1})
            cat_df['property_damage'] = cat_df['property_damage'].map({'NO': 0, 'YES': 1})
            cat_df['police_report_available'] = cat_df['police_report_available'].map({'NO': 0, 'YES': 1})
            
            try:
                # code block for training
                cat_df['fraud_reported'] = cat_df['fraud_reported'].map({'N': 0, 'Y': 1})
                cols_to_drop=['policy_csl', 'insured_education_level', 'incident_severity', 'insured_sex',
                                            'property_damage', 'police_report_available', 'fraud_reported']
            except:
                # code block for Prediction
                cols_to_drop = ['policy_csl', 'insured_education_level', 'incident_severity', 'insured_sex',
                                     'property_damage', 'police_report_available']
            # Using the dummy encoding to encode the categorical columns to numerical ones

            for col in cat_df.drop(columns=cols_to_drop).columns:
                cat_df = pd.get_dummies(cat_df, columns=[col], prefix=[col], drop_first=True)

            data.drop(columns=data.select_dtypes(include=['object']).columns, inplace=True)
            data= pd.concat([cat_df, data],axis=1)
            logging.info('encoding for categorical values successful. \
                         Exited the encode_categorical_columns method of the Preprocessor class')
            return data

        except Exception as e:
            raise FraudDetectionException(e,sys) from e
            


    def separate_label_feature(self, data):
        """
        Method Name: separate_label_feature
        Description: This method separates the features and a Label Coulmns.
        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
        On Failure: Raise Exception"""
        
        try:
            logging.info('Entered the separate_label_feature method of the Preprocessor class')

            json_path=self.data_validation_artifact.schema_file_path
            json_info= read_json_file(json_path)
            label_column=json_info['target_column']

            X=data.drop(labels=label_column,axis=1) 
            # drop the columns specified and separate the feature columns
            Y=data[label_column] # Filter the Label columns
            logging.info('Label Separation Successful.\
                          Exited the separate_label_feature method of the Preprocessor class')
            return X, Y
        
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
            '''
        

      
        
    def initiate_data_preprocessing(self)->DataPreprocessingArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()


            logging.info(f"Obtaining training and test file path.")
            file_path = self.data_ingestion_artifact.data_file_path
            
            

            schema_file_path = self.data_validation_artifact.schema_file_path
            
            logging.info(f"Loading data as pandas dataframe.")
            data_df = load_data(file_path=file_path, schema_file_path=schema_file_path)
            
            schema = read_json_file(file_path=schema_file_path)

            target_column_name = schema[TARGET_COLUMN_KEY]


            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_data_df = data_df.drop(columns=[target_column_name],axis=1)
            target_feature_data_df = data_df[target_column_name]

          
            logging.info(f"Applying preprocessing object on dataframe")
            input_feature_data_arr=preprocessing_obj.fit_transform(input_feature_data_df)
            
            data_arr = np.c_[ input_feature_data_arr, np.array(target_feature_data_df)]
           
            transformed_data_dir = self.data_transformation_config.transformed_data_dir
            

            data_file_name = os.path.basename(data_file_path).replace(".csv",".npz")
            

            transformed_data_file_path = os.path.join(transformed_data_dir, data_file_name)
            

            logging.info(f"Saving transformed data array.")
            
            save_numpy_array_data(file_path=transformed_data_file_path,array=data_arr)
           

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_data_file_path=transformed_data_file_path,
            preprocessed_object_file_path=preprocessing_obj_file_path

            )
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise FraudDetectionException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data preprocessing log completed.{'<<'*30} \n\n")