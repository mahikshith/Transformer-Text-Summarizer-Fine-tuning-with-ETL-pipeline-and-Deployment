from src.Text_summarizer.components.data_transformation import DataTransformation
from src.Text_summarizer.config.configuration import ConfigurationManager
from src.Text_summarizer.logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def data_transformation(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert_data()