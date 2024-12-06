from src.Text_summarizer.constants import *
from src.Text_summarizer.utils.common import read_yaml, create_directories
from src.Text_summarizer.entity import DataIngestionConfig , DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig

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


    def get_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        # root directory
        create_directories([config.root_dir]) 
        # return data trnsformation config 
        data_transformation_config = DataTransformationConfig(
                root_dir = config.root_dir,
                data_path =  config.data_path,
                tokenizer_name =  config.tokenizer_name
        )
        return data_transformation_config 

        # trainer 

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments
        # root directory
        create_directories([config.root_dir])

        # model trainer config
        model_trainer_config = ModelTrainerConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            base_model = config.base_model,
            per_device_train_batch_size = params.per_device_train_batch_size,
            per_device_eval_batch_size = params.per_device_eval_batch_size,
            num_train_epochs = params.num_train_epochs,
            learning_rate = params.learning_rate,
            weight_decay = params.weight_decay,
            logging_steps = params.logging_steps,
            evaluation_strategy = params.evaluation_strategy,
            eval_steps = params.eval_steps,
            save_steps = params.save_steps,
            gradient_accumulation_steps = params.gradient_accumulation_steps,
            gradient_checkpointing = params.gradient_checkpointing,
            fp16 = params.fp16,
            report_to = params.report_to
        )
        return model_trainer_config

        # Evaluation : 

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        # root directory    
        create_directories([config.root_dir]) 
        # return data ingestion config
        model_evaluation_config = ModelEvaluationConfig(
            root_dir = config.root_dir,
            data_path = config.data_path,
            model_path = config.model_path,
            tokenizer_path = config.tokenizer_path,
            metric_file = config.metric_file
        )
        return model_evaluation_config

    