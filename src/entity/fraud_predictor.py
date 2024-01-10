import os
import sys
from src.exception import FraudDetectionException
from src.util.util import load_object, read_json_file, scale_numerical_columns
import pandas as pd


from src.constant import *

from src.logger import logging
from src.util.util import  read_json_file, scale_numerical_columns, load_object, find_correct_model_file



class InsuranceData:

    def __init__(self,
                 months_as_customer: float,
                 policy_csl:str,
                 policy_deductable:float,
                 policy_annual_premium:float,
                 umbrella_limit:float,
                 insured_sex:str,
                 insured_education_level:str,
                 insured_occupation:str,
                 insured_relationship:str,
                 capitalgains:float,
                 capitalloss:float,
                 incident_type:str,
                 collision_type:str,
                 incident_severity:str,
                 authorities_contacted:str,
                 incident_hour_of_the_day:float,
                 number_of_vehicles_involved:float,
                 property_damage:str,
                 bodily_injuries:float,
                 witnesses:float,
                 police_report_available:str,
                 injury_claim:float,
                 property_claim:float,
                 vehicle_claim:float,
                 fraud_reported: str = None
                 ):
        try:
            self.months_as_customer = months_as_customer
            self.policy_csl = policy_csl
            self.policy_deductable = policy_deductable
            self.policy_annual_premium = policy_annual_premium
            self.umbrella_limit = umbrella_limit
            self.insured_sex = insured_sex
            self.insured_education_level = insured_education_level
            self.insured_occupation = insured_occupation
            self.insured_relationship = insured_relationship
            self.capitalgains=capitalgains
            self.capitalloss=capitalloss
            self.incident_type=incident_type
            self.collision_type=collision_type
            self.incident_severity=incident_severity
            self.authorities_contacted=authorities_contacted
            self.incident_hour_of_the_day=incident_hour_of_the_day
            self.number_of_vehicles_involved=number_of_vehicles_involved
            self.property_damage=property_damage
            self.bodily_injuries=bodily_injuries
            self.witnesses=witnesses
            self.police_report_available=police_report_available
            self.injury_claim=injury_claim
            self.property_claim=property_claim
            self.vehicle_claim=vehicle_claim
            
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def get_insurance_input_data_frame(self):

        try:
            housing_input_dict = self.get_insurance_data_as_dict()
            return pd.DataFrame(housing_input_dict)
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def get_insurance_data_as_dict(self):
        try:
            input_data = {
                "months_as_customer": [self.months_as_customer],
                "policy_csl": [self.policy_csl],
                "policy_deductable": [self.policy_deductable],
                "policy_annual_premium": [self.policy_annual_premium],
                "umbrella_limit": [self.umbrella_limit],
                "insured_sex": [self.insured_sex],
                "insured_education_level": [self.insured_education_level],
                "insured_occupation": [self.insured_occupation],
                "insured_relationship": [self.insured_relationship],
                "capital-gains": [self.capitalgains],
                "capital-loss": [self.capitalloss],
                "incident_type": [self.incident_type],
                "collision_type": [self.collision_type],
                "incident_severity": [self.incident_severity],
                "authorities_contacted": [self.authorities_contacted],
                "incident_hour_of_the_day": [self.incident_hour_of_the_day],
                "number_of_vehicles_involved": [self.number_of_vehicles_involved],
                "property_damage": [self.property_damage],
                "bodily_injuries": [self.bodily_injuries],
                "witnesses": [self.witnesses],
                "police_report_available": [self.police_report_available],
                "injury_claim": [self.injury_claim],
                "property_claim": [self.property_claim],
                "vehicle_claim": [self.vehicle_claim],
                  
                }
            return input_data
        except Exception as e:
            raise FraudDetectionException(e, sys)








class FraudPredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
        


    def predict(self, data):
        try:

            predictions=[]

            ohe_columns= ["insured_occupation", "insured_relationship", "incident_type", "collision_type", "authorities_contacted"]
            encoded_data = pd.get_dummies(data, columns=ohe_columns)
            dump_col=['authorities_contacted_Ambulance','authorities_contacted_None', 
                'collision_type_Front Collision', 'incident_type_Multi-vehicle Collision', 
                'insured_occupation_adm-clerical', 'insured_relationship_husband']
            
            encoded_data=encoded_data.astype(float)

            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            cluster_object_file_path =os.path.join(latest_model_dir, 'KMeans')
            file_name = os.listdir(cluster_object_file_path)[0]
            cluster_model = os.path.join(cluster_object_file_path,file_name) 
            kmeans=load_object(cluster_model)
            columns_used_for_clustering = kmeans.columns_used
            col_to_add=columns_used_for_clustering.difference(encoded_data.columns)
            for col in col_to_add:
                encoded_data[col] = 0

            dataframe = encoded_data.drop(columns=[col for col in dump_col if col in encoded_data.columns])

            df=dataframe[columns_used_for_clustering]


            df.astype(float) 
            cluster=kmeans.predict(df)

            logging.info(f"got cluster")

            model_name = find_correct_model_file(path=latest_model_dir, cluster_number=cluster[0])
            model_dir = os.path.join(latest_model_dir, model_name)
            file_name = os.listdir(model_dir)[0]
            
            model = load_object(os.path.join(model_dir, file_name))
            result=(model.predict(df))
            if result==0:
                predictions.append('N')
            else:
                predictions.append('Y')

            return predictions
            
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

        



