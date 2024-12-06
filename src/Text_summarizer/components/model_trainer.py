from transformers import pipeline , set_seed
import os
from datasets import load_dataset , load_from_disk
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.Text_summarizer.entity import ModelTrainerConfig

from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments , Trainer
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'




class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config 

    def train(self):    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.base_model).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        # loading data 
        dataset_samsum = load_from_disk(self.config.data_path)
        # trainer
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=500,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            fp16=self.config.fp16,
            report_to=self.config.report_to
        )
        
        trainer_args.data_collator = seq2seq_data_collator

        # use only 1000 samples for training 

        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum["train"].shuffle().select(range(1000)),
            eval_dataset=dataset_samsum["validation"])

        trainer.train()
        # SAVE MODEL
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"Trained-pegasus-model"))

        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"new-tokenizer"))