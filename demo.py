from src.pipeline.pipeline import Pipeline
#from src.pipeline.pipe import Pipeline
from src.exception import FraudDetectionException
from src.config.configuration import Configuartion
from src.logger import logging
from src.constant import CONFIG_DIR, get_current_time_stamp


def main():
    try:
        pipeline = Pipeline(config=Configuartion(current_time_stamp=get_current_time_stamp()))
        #pipeline.run_pipeline()

       # pipeline.start()
        pipeline.initiate_bulk_prediction()



    except Exception as e:
        logging.error(f"{e}")
        print(e)



if __name__=="__main__":
    main()