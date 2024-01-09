import sys,os
import numpy as np
#from cgi import test
from sklearn import preprocessing
from src.exception import FraudDetectionException
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig 
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
from src.constant import *
from src.util.util import  read_json_file, save_data, save_object, load_data, save_model





from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from os import listdir



import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator


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

         # Mapping for 'fraud reported'
        fraud_mapping={'N': 0, 'Y': 1}
        X_mapped['fraud_reported'] = X_mapped['fraud_reported'].map(fraud_mapping)

        return X_mapped



class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        


    def remove_unwanted_spaces(self, data)-> pd.DataFrame:
        """
        This method removes the unwanted spaces from a pandas dataframe.
        Output: A pandas DataFrame after removing the spaces.
        On Failure: Raise Exception"""
        try:
            ''' logging.info("Loading data frame from the provided dataset")
            raw_data_dir = self.data_ingestion_artifact.data_file_path
            file_name = os.listdir(raw_data_dir)[0]
            file_path = os.path.join(raw_data_dir,file_name)

            logging.info(f"Reading csv file: [{file_path}]")
            data=pd.read_csv(file_path)'''

            data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            logging.info("Removed unwanted spaces in the dataframe")

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
            # Dropping the columns with all the null values
            null_columns = data.columns[data.isnull().all()]
            data.drop(null_columns, axis=1, inplace=True)

            json_path=self.data_validation_artifact.schema_file_path
            json_info= read_json_file(json_path)
            columns=json_info['columns_to_remove']
         
            useful_data=data.drop(labels=columns, axis=1) 
            # drop the labels specified in the columns

            useful_data.replace('?',np.NaN,inplace=True)
            logging.info('Removing certain columns from the dataframe and replacing ? with Null values')
           
            return useful_data
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        

    def impute_missing_values(self, data)-> pd.DataFrame:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_json_file(file_path=schema_file_path)

            numerical_columns = [x for x in dataset_schema[NUMERICAL_COLUMN_KEY] 
                                 if x not in dataset_schema[COLUMNS_TO_REMOVE] ]
            categorical_columns = [x for x in dataset_schema[CATEGORICAL_COLUMN_KEY] 
                                   if x not in dataset_schema[COLUMNS_TO_REMOVE] ]
            logging.info('Imputing numerical values')
            num_imputer = SimpleImputer(strategy='mean')
            data[numerical_columns] = num_imputer.fit_transform(data[numerical_columns])
            
            logging.info('Imputing categorical values')
            cat_imputer = SimpleImputer(strategy='most_frequent')
            data[categorical_columns] = cat_imputer.fit_transform(data[categorical_columns])

            return data

        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        


    def encode_data(self, data)->pd.DataFrame:
        try:
            custom_encoding_obj=CustomEncoder()
            logging.info('Cusom encoding the data')
            data=custom_encoding_obj.fit_transform(data)

            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_json_file(file_path=schema_file_path)
            ohe_columns=dataset_schema[COLUMNS_FOR_OHE]
            logging.info('One Hot Eencoding for remaining categorical data')
            encoded_data = pd.get_dummies(data, columns=ohe_columns, drop_first=True)

            return encoded_data

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
            label_column=json_info[TARGET_COLUMN_KEY]

            X=data.drop(labels=label_column,axis=1) 
            # drop the columns specified and separate the feature columns
            Y=data[label_column] # Filter the Label columns
            logging.info('Label Separation Successful.\
                          Exited the separate_label_feature method of the Preprocessor class')
            return X, Y
        
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
            
    def elbow_plot(self, data):
        """
        Method Name: elbow_plot
        Description: This method saves the plot to decide the optimum number of clusters to the file.
        Output: A picture saved to the directory

        """
        try:
            
            logging.info( 'Entered the elbow_plot method of the KMeansClustering class')
            wcss=[] # initializing an empty list
        
            for i in range (1,11):
                # initializing the KMeans object
                kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42) 
                kmeans.fit(data) # fitting the data to the KMeans Algorithm
                wcss.append(kmeans.inertia_)
            plt.plot(range(1,11),wcss) # creating the graph between WCSS and the number of clusters
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            #plt.show()
            img_name='K-Means_Elbow.PNG'
            path = self.data_transformation_config.cluster_object_file_path
            img_file_path = os.path.join(path, img_name)

            #plt.savefig(img_file_path) # saving the elbow plot locally
            # finding the value of the optimum cluster programmatically
            kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            logging.info('The optimum number of clusters is: '+str(kn.knee)+' \
                         . Exited the elbow_plot method of the KMeansClustering class')
            
            return kn.knee
        
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        

    def create_clusters(self,data,number_of_clusters):
        """
        Method Name: create_clusters
        Description: Create a new dataframe consisting of the cluster information.
        Output: A datframe with cluster column
        """
        logging.info('Entered the create_clusters method of the KMeansClustering class')
        
        try:
            kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
            #self.data = self.data[~self.data.isin([np.nan, np.inf, -np.inf]).any(1)]
            y_kmeans=kmeans.fit_predict(data) #  divide data into clusters

            path=self.data_transformation_config.cluster_object_file_path
            
            save_model(path, kmeans, 'KMeans') 
            # saving the KMeans model to directory
            # passing 'Model' as the functions need three parameters

            data['Cluster']=y_kmeans  # create a new column in dataset for storing the cluster information
            logging.info( 'succesfully created '+str(number_of_clusters)+ 'clusters. Exited the create_clusters method of the KMeansClustering class')
            return data
        
        except Exception as e:
            raise FraudDetectionException(e,sys) from e
        
          
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"Obtaining data file path.")
            schema_file_path = self.data_validation_artifact.schema_file_path
            data_file_path = self.data_ingestion_artifact.data_file_path
            logging.info(f"Loading data as pandas dataframe.")

            #data_df = load_data(path=data_file_path, schema_file_path=schema_file_path)

            file_name = os.listdir(data_file_path)[0]
            file_path = os.path.join(data_file_path,file_name)

            data_df = pd.read_csv(file_path)

            schema = read_json_file(file_path=schema_file_path)
            target_column_name = schema[TARGET_COLUMN_KEY]

            space_removed_data= self.remove_unwanted_spaces(data_df)
            columns_removed_data=self.remove_columns(space_removed_data)

            logging.info('Imputing missing values')
            imputed_data=self.impute_missing_values(columns_removed_data)

            logging.info('Encoding categorical values')
            encoded_data=self.encode_data(imputed_data)

            input_feature_data_df = encoded_data.drop(columns=target_column_name,axis=1)
            target_feature_data_df = encoded_data[target_column_name]

            clusters=self.elbow_plot(input_feature_data_df)
            clustered_data=self.create_clusters(input_feature_data_df, clusters)
            clustered_data['Labels']=target_feature_data_df

            #convert the clustered data into float
            clustered_data=clustered_data.astype(float) 
            transformed_data_dir = self.data_transformation_config.transformed_data_dir
            data_file_name = 'Transformed_dataframe.csv'

            transformed_data_file_path = os.path.join(transformed_data_dir, data_file_name)
            logging.info(f"Saving transformed data.")
            save_data(transformed_data_file_path, clustered_data)

                 
            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",

            transformed_data_file_path=transformed_data_dir, 






            
            cluster_object_file_path=self.data_transformation_config.cluster_object_file_path,
            number_of_clusters=clusters
            )

            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise FraudDetectionException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data preprocessing log completed.{'<<'*30} \n\n")