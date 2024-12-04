from src.Text_summarizer.constants import *
from src.Text_summarizer.utils.common import read_yaml, create_directories
from src.Text_summarizer.entity import DataIngestionConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        # root directory
        create_directories([config.root_dir]) 
        # return data ingestion config 
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_url = config.source_url,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )
        return data_ingestion_config 