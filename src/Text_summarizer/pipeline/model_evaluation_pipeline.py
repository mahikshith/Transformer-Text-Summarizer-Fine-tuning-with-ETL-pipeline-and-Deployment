from src.Text_summarizer.config.configuration import ConfigurationManager
from src.Text_summarizer.components.model_evaluation import ModelEvaluation
from src.Text_summarizer.logging import logger

class ModelEvaluationPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config

    def model_evaluation(self):
        model_evaluation_config = self.config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate()

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def model_evaluation(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate()