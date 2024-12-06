from src.Text_summarizer.components.model_trainer import ModelTrainer
from src.Text_summarizer.config.configuration import ConfigurationManager
from src.Text_summarizer.logging import logger 


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def model_trainer(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_train = ModelTrainer(config=model_trainer_config)
        model_train.train()