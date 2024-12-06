from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir : Path
    source_url : Path
    local_data_file : Path
    unzip_dir : Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path

# trainer
@dataclass
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    base_model: Path
    # params
    per_device_train_batch_size : int
    per_device_eval_batch_size : int  
    num_train_epochs : int             
    learning_rate : float       
    weight_decay : float
    logging_steps : int
    evaluation_strategy : str
    eval_steps : int
    save_steps : float
    gradient_accumulation_steps : int
    gradient_checkpointing : bool
    fp16 :bool              
    report_to : str  
