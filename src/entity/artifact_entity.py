from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact", ["data_file_path", "is_ingested", "message"])

DataValidationArtifact = namedtuple("DataValidationArtifact",
["schema_file_path","report_file_path","report_page_file_path","is_validated","message"])


DataTransformationArtifact = namedtuple("DataTransformationArtifact", 
                                        ["is_transformed", "message", "transformed_data_file_path", 
                                         "cluster_object_file_path", "number_of_clusters"])


ModelTrainerArtifact = namedtuple("ModelTrainerArtifact", ["is_trained", "message", "trained_model_file_path",
                                                            "train_accuracy", "test_accuracy", "model_accuracy"])

ModelPusherArtifact = namedtuple("ModelPusherArtifact", ["is_model_pusher", "export_model_file_path"])

DataPredictionArtifact = namedtuple("PredictionArtifact", ["is_predicted", "export_data_file_path"])