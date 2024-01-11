from flask import Flask, request
import sys

import pip
from src.util.util import read_yaml_file, write_yaml_file
from matplotlib.style import context
from src.logger import logging
from src.exception import FraudDetectionException
import os, sys
import json
from src.config.configuration import Configuartion
from src.constant import CONFIG_DIR, get_current_time_stamp
from src.pipeline.pipeline import Pipeline
from src.entity.fraud_predictor import FraudPredictor, InsuranceData
from flask import send_file, abort, render_template


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "src"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


from src.logger import get_log_dataframe

INSURANCE_DATA_KEY = "insurance_data"
FRAUD_VALUE_KEY = "fraud_value"

app = Flask(__name__)


@app.route('/artifact', defaults={'req_path': 'src'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("src", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuartion(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message }
    
    return render_template('train.html', context=context)

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    context = {
        INSURANCE_DATA_KEY: None,
        FRAUD_VALUE_KEY: None }

    if request.method == 'POST':

        logging.info(f"post started")
        months_as_customer=float(request.form['months_as_customer'])
        logging.info(f"post started 2")
        policy_csl=float(request.form['policy_csl'])
        policy_deductable=float(request.form['policy_deductable'])
        policy_annual_premium=float(request.form['policy_annual_premium'])
        umbrella_limit=float(request.form['umbrella_limit'])
        insured_sex=float(request.form['insured_sex'])
        insured_education_level=float(request.form['insured_education_level'])
        insured_occupation=str(request.form['insured_occupation'])
        insured_relationship=str(request.form['insured_relationship'])
        capitalgains=float(request.form['capitalgains'])
        capitalloss=float(request.form['capitalloss'])
        incident_type=str(request.form['incident_type'])
        collision_type=str(request.form['collision_type'])
        incident_severity=float(request.form['incident_severity'])
        authorities_contacted=str(request.form['authorities_contacted'])
        incident_hour_of_the_day=float(request.form['incident_hour_of_the_day'])
        number_of_vehicles_involved=float(request.form['number_of_vehicles_involved'])
        property_damage=float(request.form['property_damage'])
        bodily_injuries=float(request.form['bodily_injuries'])
        witnesses=float(request.form['witnesses'])
        police_report_available=float(request.form['police_report_available'])
        injury_claim=float(request.form['injury_claim'])
        property_claim=float(request.form['property_claim'])
        vehicle_claim=float(request.form['vehicle_claim'])

        insurance_data=InsuranceData(months_as_customer=months_as_customer,
                                     policy_csl=policy_csl,
                                     policy_deductable=policy_deductable,
                                     policy_annual_premium=policy_annual_premium,
                                     umbrella_limit=umbrella_limit,
                                     insured_sex=insured_sex,
                                     insured_education_level=insured_education_level,
                                     insured_occupation=insured_occupation,
                                     insured_relationship=insured_relationship,
                                     capitalgains=capitalgains,
                                     capitalloss=capitalloss,
                                     incident_type=incident_type,
                                     collision_type=collision_type,
                                     incident_severity=incident_severity,
                                     authorities_contacted=authorities_contacted,
                                     incident_hour_of_the_day=incident_hour_of_the_day,
                                     number_of_vehicles_involved=number_of_vehicles_involved,
                                     property_damage=property_damage,
                                     bodily_injuries=bodily_injuries,
                                     witnesses=witnesses,
                                     police_report_available=police_report_available,
                                     injury_claim=injury_claim,
                                     property_claim=property_claim,
                                     vehicle_claim=vehicle_claim
                                     )

        insurance_df = insurance_data.get_insurance_input_data_frame()
        insurance_predictor = FraudPredictor(model_dir=MODEL_DIR)
        fraud_value=insurance_predictor.predict(insurance_df)
        
        context = {
            INSURANCE_DATA_KEY:insurance_data.get_insurance_data_as_dict(),
            FRAUD_VALUE_KEY:fraud_value}
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)



@app.route('/bulkpredict', methods=['GET', 'POST'])
def bulkpredict():
    message = ""
    pipeline = Pipeline(config=Configuartion(current_time_stamp=get_current_time_stamp()))
    pipeline.initiate_bulk_prediction()



























@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except  Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
    app.run()
