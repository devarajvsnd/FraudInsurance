from src.logger import logging
from src.exception import FraudDetectionException
from src.entity.artifact_entity import ModelPusherArtifact, DataTransformationArtifact, ModelTrainerArtifact 
from src.entity.config_entity import ModelPusherConfig
import os, sys
import shutil


class ModelPusher:

    def __init__(self, model_pusher_config: ModelPusherConfig, model_trainer_artifact: ModelTrainerArtifact,
                 data_trans_artifact: DataTransformationArtifact
                 ):
        try:
            logging.info(f"{'>>' * 30}Model Pusher log started.{'<<' * 30} ")
            self.model_pusher_config = model_pusher_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_trans_artifact=data_trans_artifact

        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def export_model(self) -> ModelPusherArtifact:
        try:
            #clusters=self.data_trans_artifact.number_of_clusters
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            cluster_model_file_path= self.data_trans_artifact.cluster_object_file_path

            export_dir = self.model_pusher_config.export_dir_path
            os.makedirs(export_dir, exist_ok=True)
            logging.info(f"Exporting model file: [{export_dir}]")

            for item in os.listdir(trained_model_file_path):
                source_item = os.path.join(trained_model_file_path, item)
                destination_item = os.path.join(export_dir, item)

                if os.path.isfile(source_item):  # If the item is a file, copy it
                    shutil.copy(source_item, destination_item)
                elif os.path.isdir(source_item):  # If the item is a directory, copy its contents recursively
                    shutil.copytree(source_item, destination_item, dirs_exist_ok=True)
                
                logging.info(f"Exported all model files of clusters")

            shutil.copytree(cluster_model_file_path, export_dir, dirs_exist_ok=True)
            logging.info(f"Exported cluster model as well")
            #we can call a function to save model to Azure blob storage/ google cloud strorage / s3 bucket
            model_pusher_artifact = ModelPusherArtifact(is_model_pusher=True,
                                                        export_model_file_path=export_dir
                                                        )
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            return model_pusher_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            return self.export_model()
        except Exception as e:
            raise FraudDetectionException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 20}Model Pusher log completed.{'<<' * 20} ")
