import os
import sys
from src.exception import FraudDetectionException
from src.util.util import load_object, read_json_file, scale_numerical_columns
import pandas as pd

from src.entity.config_entity import DataValidationConfig

from src.logger import logging



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
            self.fraud_reported=fraud_reported
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
                "fraud_reported": [self.fraud_reported]   
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

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            #check model for individual cluster
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            median_house_value = model.predict(X)
            return median_house_value
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
        



