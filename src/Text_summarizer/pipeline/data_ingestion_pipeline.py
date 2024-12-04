from src.Text_summarizer.components.data_ingestion import DataIngestion
from src.Text_summarizer.config.configuration import ConfigurationManager
from src.Text_summarizer.logging import logger

# data ingestion training pipeline

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def data_ingestion(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
