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
            columns=json_info['columns_to remove']
         
            useful_data=data.drop(labels=columns, axis=1) 
            # drop the labels specified in the columns

            data.replace('?',np.NaN,inplace=True)
           
            return useful_data
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
            null_present = False
            cols_with_missing_values=[]
            cols = data.columns
        
            null_counts=data.isna().sum() # check for the count of null values per column

            for i in range(len(null_counts)):
                if null_counts[i]>0:
                    null_present=True
                    cols_with_missing_values.append(cols[i])

            if(null_present): # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())

                dataframe_with_null.to_csv('preprocessing_data/null_values.csv') 
                # storing the null column information to file

            logging.info('Finding missing values is a success.Data written to the null values file\
                         . Exited the is_null_present method of the Preprocessor class')
            
            return null_present, cols_with_missing_values
        
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
                data= data
                cols_with_missing_values=cols_with_missing_values
            
                imputer = CategoricalImputer()
                for col in cols_with_missing_values:
                    data[col] = imputer.fit_transform(data[col])
                logging.info('Imputing missing values Successful.\
                            Exited the impute_missing_values method of the Preprocessor class')
                return data
            
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
    def handle_imbalanced_dataset(self,x,y):
        """
        Method Name: handle_imbalanced_dataset
        Description: This method handles the imbalanced dataset to make it a balanced one.
        Output: new balanced feature and target columns
        On Failure: Raise Exception"""

        try:
            logging.info('Entered the handle_imbalanced_dataset method of the Preprocessor class')
            rdsmple = RandomOverSampler()
            x_sampled,y_sampled  = rdsmple.fit_sample(x,y)
            logging.info('dataset balancing successful. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            return x_sampled, y_sampled

        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        '''

  
        

    def scale_numerical_columns(self,data):
        """
        Method Name: scale_numerical_columns
        Description: This method scales the numerical values using the Standard scaler.
        Output: A dataframe with scaled values
        On Failure: Raise Exception """
        
        try:
            logging.info('Entered the scale_numerical_columns method of the Preprocessor class')
            self.data=data
            json_path=self.data_validation_artifact.schema_file_path
            json_info= read_json_file(json_path)
            num_df=[json_info['numerical_columns']]
     
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(num_df)
            scaled_num_df = pd.DataFrame(data=scaled_data, columns=num_df.columns,index=data.index)
            data.drop(columns=scaled_num_df.columns, inplace=True)
            
            data = pd.concat([scaled_num_df, data], axis=1)

            logging.info('scaling for numerical values successful.\
                          Exited the scale_numerical_columns method of the Preprocessor class')
            return data

        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        

    
        




    def initiate_data_preprocessing(self)->DataPreprocessingArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()


            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            

            schema_file_path = self.data_validation_artifact.schema_file_path
            
            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            
            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)

            schema = read_json_file(file_path=schema_file_path)

            target_column_name = schema[TARGET_COLUMN_KEY]


            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessed_object_file_path=preprocessing_obj_file_path

            )
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise FraudDetectionException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data preprocessing log completed.{'<<'*30} \n\n")