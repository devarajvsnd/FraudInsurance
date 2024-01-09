from collections import namedtuple


DataIngestionConfig=namedtuple("DataIngestionConfig", ["dataset_download_url","tgz_download_dir","raw_data_dir"])


DataValidationConfig = namedtuple("DataValidationConfig", ["schema_file_path","report_file_path","report_page_file_path"])


DataTransformationConfig = namedtuple("DataTransformationConfig", ["transformed_data_dir", "cluster_object_file_path"])


ModelTrainerConfig = namedtuple("ModelTrainerConfig", ["trained_model_file_path","base_accuracy","model_config_file_path"])



#ModelEvaluationConfig = namedtuple("ModelEvaluationConfig", ["model_evaluation_file_path","time_stamp"])


ModelPusherConfig = namedtuple("ModelPusherConfig", ["export_dir_path"])

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])

PredictionConfig =namedtuple("PredictionConfig", ["dataset_download_url","predicted_path", "tgz_download_dir","raw_data_dir"])