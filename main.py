from src.Text_summarizer.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.Text_summarizer.logging import logger
from src.Text_summarizer.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.Text_summarizer.pipeline.model_trainer_pipeline import ModelTrainerTrainingPipeline
from src.Text_summarizer.pipeline.model_evaluation_pipeline import ModelEvaluationTrainingPipeline


# data ingestion training pipeline 
"""
try:
    logger.info(f"stage Data Ingestion Stage started ")
    data_ingestion_training_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_training_pipeline.data_ingestion()
    logger.info(f"stage Data Ingestion Stage completed ")
except Exception as e:
    logger.exception(e)
    raise e

# data transformation training pipeline

try:
    logger.info(f"stage Data Transformation Stage started ")
    data_transformation_training_pipeline = DataTransformationTrainingPipeline()
    data_transformation_training_pipeline.data_transformation()
    logger.info(f"stage Data Transformation Stage completed ")
except Exception as e:
    logger.exception(e)
    raise e


# model trainer training pipeline

stage_name = "Model Trainer Stage"
try:
    logger.info(f"stage {stage_name} started ")
    model_trainer_training_pipeline = ModelTrainerTrainingPipeline()
    model_trainer_training_pipeline.model_trainer()
    logger.info(f"stage {stage_name} completed ")
except Exception as e:
    logger.exception(e)
    raise e
"""
# model evaluation training pipeline

stage_name = "Model Evaluation Stage"
try:
    logger.info(f"stage {stage_name} started ")
    model_evaluation_training_pipeline = ModelEvaluationTrainingPipeline()
    model_evaluation_training_pipeline.model_evaluation()
    logger.info(f"stage {stage_name} completed ")
except Exception as e:
    logger.exception(e)
    raise e