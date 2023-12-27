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
from src.util.util import read_yaml_file,save_object,save_numpy_array_data,load_data

from sklearn_pandas import CategoricalImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from os import listdir

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
        

    def remove_unwanted_spaces(self):
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
            data= pd.read_csv(file_path)

            logging.info("Entered the remove_unwanted_spaces method of the Preprocessor class")
            data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


        except Exception as e:
            raise FraudDetectionException(e,sys) from e


    def replaceMissingWithNull(self):
        """
        Method Name: replaceMissingWithNull
        Description: This method replaces the missing values in columns with "NULL" to
                    store in the table. We are using substring in the first column to
                    keep only "Integer" data for ease up the loading.
                    This column is anyways going to be removed during training."""
        try:
            logging.info("TEntered the replaceMissingWithNull method of the Preprocessor class")


            onlyfiles = [f for f in listdir(self.goodDataPath)]

            for file in onlyfiles:
                data = pd.read_csv(self.goodDataPath + "/" + file)
                # list of columns with string datatype variables
                columns = ["policy_bind_date","policy_state","policy_csl","insured_sex","insured_education_level",
                           "insured_occupation","insured_hobbies", "insured_relationship","incident_state",
                           "incident_date","incident_type", "collision_type","incident_severity",
                           "authorities_contacted","incident_city", "incident_location","property_damage",
                           "police_report_available","auto_make","auto_model", "fraud_reported"]

                for col in columns:
                        data[col] = data[col].apply(lambda x: "'" + str(x) + "'")

                data.to_csv(self.goodDataPath + "/" + file, index=None, header=True)
                logging.info(" %s: Quotes added successfully!!" % file)
               #log_file.write("Current Date :: %s" %date +"\t" + "Current time:: %s" % current_time + "\t \t" +  + "\n")
          
        except Exception as e:
            raise FraudDetectionException(e,sys) from e


    def remove_columns(self,data,columns):
        """
        Method Name: remove_columns
        Description: This method removes the given columns from a pandas dataframe.
        Output: A pandas DataFrame after removing the specified columns.
        On Failure: Raise Exception"""
        
        try:
            logging.info('Entered the remove_columns method of the Preprocessor class')
            self.data=data
            self.columns=columns
        
            useful_data=self.data.drop(labels=self.columns, axis=1) 
            # drop the labels specified in the columns
           
            return useful_data
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        

    def separate_label_feature(self, data, label_column_name):
        """
        Method Name: separate_label_feature
        Description: This method separates the features and a Label Coulmns.
        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
        On Failure: Raise Exception"""
        
        try:
            logging.info('Entered the separate_label_feature method of the Preprocessor class')
        
            self.X=data.drop(labels=label_column_name,axis=1) 
            # drop the columns specified and separate the feature columns
            self.Y=data[label_column_name] # Filter the Label columns
            logging.info('Label Separation Successful.\
                          Exited the separate_label_feature method of the Preprocessor class')
            return self.X,self.Y
        
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        


    def is_null_present(self,data):
        """
        Method Name: is_null_present
        Description: This method checks whether there are null values present in the pandas Dataframe or not.
        Output: Returns True if null values are present in the DataFrame, False if they are not present and
                returns the list of columns for which null values are present.
        On Failure: Raise Exception"""

        try:
            logging.info('Entered the is_null_present method of the Preprocessor class')
            self.null_present = False
            self.cols_with_missing_values=[]
            self.cols = data.columns
        
            self.null_counts=data.isna().sum() # check for the count of null values per column

            for i in range(len(self.null_counts)):
                if self.null_counts[i]>0:
                    self.null_present=True
                    self.cols_with_missing_values.append(self.cols[i])
            if(self.null_present): # write the logs to see which columns have null values
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null['columns'] = data.columns
                self.dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                self.dataframe_with_null.to_csv('preprocessing_data/null_values.csv') 
                # storing the null column information to file

            logging.info('Finding missing values is a success.Data written to the null values file\
                         . Exited the is_null_present method of the Preprocessor class')
            
            return self.null_present, self.cols_with_missing_values
        
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        

    def impute_missing_values(self, data, cols_with_missing_values):
        """
        Method Name: impute_missing_values
        Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
        Output: A Dataframe which has all the missing values imputed.
        On Failure: Raise Exception """
        
        try:
            logging.info('Entered the impute_missing_values method of the Preprocessor class')
            self.data= data
            self.cols_with_missing_values=cols_with_missing_values
        
            self.imputer = CategoricalImputer()
            for col in self.cols_with_missing_values:
                self.data[col] = self.imputer.fit_transform(self.data[col])
            logging.info('Imputing missing values Successful.\
                          Exited the impute_missing_values method of the Preprocessor class')
            return self.data
        
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        

    def scale_numerical_columns(self,data):
        """
        Method Name: scale_numerical_columns
        Description: This method scales the numerical values using the Standard scaler.
        Output: A dataframe with scaled values
        On Failure: Raise Exception """
        
        try:
            logging.info('Entered the scale_numerical_columns method of the Preprocessor class')
            self.data=data
            self.num_df = self.data[['months_as_customer', 'policy_deductable', 'umbrella_limit',
                            'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
                            'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'injury_claim',
                            'property_claim',
                            'vehicle_claim']]
      
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.num_df)
            self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=self.num_df.columns,index=self.data.index)
            self.data.drop(columns=self.scaled_num_df.columns, inplace=True)
            self.data = pd.concat([self.scaled_num_df, self.data], axis=1)

            logging.info('scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return self.data

        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        

    def encode_categorical_columns(self,data):
        """
        Method Name: encode_categorical_columns
        Description: This method encodes the categorical values to numeric values.
        Output: dataframe with categorical values converted to numerical values
        On Failure: Raise Exception"""
            
        try:    
            logging.info('Entered the encode_categorical_columns method of the Preprocessor class')
            self.data=data
        
            self.cat_df = self.data.select_dtypes(include=['object']).copy()
            self.cat_df['policy_csl'] = self.cat_df['policy_csl'].map({'100/300': 1, '250/500': 2.5, '500/1000': 5})
            self.cat_df['insured_education_level'] = self.cat_df['insured_education_level'].map(
                {'JD': 1, 'High School': 2, 'College': 3, 'Masters': 4, 'Associate': 5, 'MD': 6, 'PhD': 7})
            self.cat_df['incident_severity'] = self.cat_df['incident_severity'].map(
                {'Trivial Damage': 1, 'Minor Damage': 2, 'Major Damage': 3, 'Total Loss': 4})
            self.cat_df['insured_sex'] = self.cat_df['insured_sex'].map({'FEMALE': 0, 'MALE': 1})
            self.cat_df['property_damage'] = self.cat_df['property_damage'].map({'NO': 0, 'YES': 1})
            self.cat_df['police_report_available'] = self.cat_df['police_report_available'].map({'NO': 0, 'YES': 1})
            
            try:
                # code block for training
                self.cat_df['fraud_reported'] = self.cat_df['fraud_reported'].map({'N': 0, 'Y': 1})
                self.cols_to_drop=['policy_csl', 'insured_education_level', 'incident_severity', 'insured_sex',
                                            'property_damage', 'police_report_available', 'fraud_reported']
            except:
                # code block for Prediction
                self.cols_to_drop = ['policy_csl', 'insured_education_level', 'incident_severity', 'insured_sex',
                                     'property_damage', 'police_report_available']
            # Using the dummy encoding to encode the categorical columns to numerical ones

            for col in self.cat_df.drop(columns=self.cols_to_drop).columns:
                self.cat_df = pd.get_dummies(self.cat_df, columns=[col], prefix=[col], drop_first=True)

            self.data.drop(columns=self.data.select_dtypes(include=['object']).columns, inplace=True)
            self.data= pd.concat([self.cat_df,self.data],axis=1)
            self.logger_object.log(self.file_object, 'encoding for categorical values successful. Exited the encode_categorical_columns method of the Preprocessor class')
            return self.data

        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
        
    def handle_imbalanced_dataset(self,x,y):
        """
        Method Name: handle_imbalanced_dataset
        Description: This method handles the imbalanced dataset to make it a balanced one.
        Output: new balanced feature and target columns
        On Failure: Raise Exception"""

        try:
            logging.info('Entered the handle_imbalanced_dataset method of the Preprocessor class')
            self.rdsmple = RandomOverSampler()
            self.x_sampled,self.y_sampled  = self.rdsmple.fit_sample(x,y)
            logging.info('dataset balancing successful. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            return self.x_sampled,self.y_sampled

        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        

