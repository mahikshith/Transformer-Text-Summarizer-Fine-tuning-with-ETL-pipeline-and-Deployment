from src.Text_summarizer.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.Text_summarizer.logging import logger


# data ingestion training pipeline 

stage_name = "Data Ingestion Stage"
try:
    logger.info(f"stage {stage_name} started ")
    data_ingestion_training_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_training_pipeline.data_ingestion()
    logger.info(f"stage {stage_name} completed ")
except Exception as e:
    logger.exception(e)
    raise e

