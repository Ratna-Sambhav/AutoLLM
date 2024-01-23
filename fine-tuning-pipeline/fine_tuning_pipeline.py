import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, LoftQConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import time
import json
import multiprocessing
from multiprocessing import set_start_method
from merge_peft_model import merge_peft_adaptors
import argparse

class llm_training_hf():

  def __init__(self, my_dict_path):

    with open(my_dict_path, 'r') as f:
      my_dict = json.load(f)

    ## Model and Dataset
    self.model_name = my_dict.get('model_name')
    self.fine_tuned_model_name = my_dict.get('fine_tuned_model_name')
    self.load_in_8bit = my_dict.get('load_in_8bit')

    ## Dataset description
    self.dataset_name = my_dict.get('dataset_name')
    self.context_name = my_dict.get('context_name')
    self.query_name = my_dict.get('query_name')
    self.response_name = my_dict.get('response_name')

    ## Tensorboard configs
    self.tensorboard_port = my_dict.get('tensorboard_port')
    self.logdir = my_dict.get('logdir')

    ## QLora Params
    self.lora = my_dict.get('lora')
    self.lora_r = my_dict.get('lora_r')
    self.lora_alpha = my_dict.get('lora_alpha')
    self.lora_dropout = my_dict.get('lora_dropout')
    self.q_lora = my_dict.get('qlora')

    ## Training Parameters
    # Output directory where the model predictions and checkpoints will be saved
    self.output_dir = my_dict.get('output_dir')
    # Number of training epochs
    self.num_train_epochs = my_dict.get('num_train_epochs')
    # fp16/bf16 training
    self.fp16 = my_dict.get('fp16')
    self.bf16 = my_dict.get('bf16')
    # Batch size per GPU for training
    self.per_device_train_batch_size = my_dict.get('per_device_train_batch_size')
    # Batch size per GPU for evaluation
    self.per_device_eval_batch_size = my_dict.get('per_device_eval_batch_size')
    # Number of update steps to accumulate the gradients for
    self.gradient_accumulation_steps = my_dict.get('gradient_accumulation_steps')
    # Enable gradient checkpointing
    self.gradient_checkpointing = my_dict.get('gradient_checkpointing')
    # Maximum gradient normal (gradient clipping)
    self.max_grad_norm = my_dict.get('max_grad_norm')
    # Initial learning rate (AdamW optimizer)
    self.learning_rate = my_dict.get('learning_rate')
    # Weight decay to apply to all layers except bias/LayerNorm weights
    self.weight_decay = my_dict.get('weight_decay')
    # Optimizer to use
    self.optim = my_dict.get('optim')
    # Learning rate schedule (constant a bit better than cosine)
    self.lr_scheduler_type = my_dict.get('lr_scheduler_type')
    # Number of training steps (overrides num_train_epochs)
    self.max_steps = my_dict.get('max_steps')
    # Ratio of steps for a linear warmup (from 0 to learning rate)
    self.warmup_ratio = my_dict.get('warmup_ratio')
    # Group sequences into batches with same length
    self.group_by_length = my_dict.get('group_by_length')
    # Save checkpoint every X updates steps
    self.save_steps = my_dict.get('save_steps')
    # Log every X updates steps
    self.logging_steps = my_dict.get('logging_steps')

    ## SFT params
    # Maximum sequence length to use
    self.max_seq_length = my_dict.get('max_seq_length')
    # Pack multiple short examples in the same input sequence to increase efficiency
    self.packing = my_dict.get('packing')
    # Load the entire model on the GPU 0
    self.device_map = my_dict.get('device_map')

  def get_formatter_function(self):
    def formatting_prompts_func(example):
      if self.context_name:
        context_key = self.context_name.split(':')[0]
        context_col = self.context_name.split(':')[1]
      if self.query_name:
        query_key = self.query_name.split(':')[0]
        query_col = self.query_name.split(':')[1]
      if self.response_name:
        response_key = self.response_name.split(':')[0]
        response_col = self.response_name.split(':')[1]
      output_texts = []
      for i in range(len(example[response_col])):
        context = f'### {context_key}: {example[context_col]}\n' if self.context_name else ''
        question = f' ### {query_key}: {example[query_col][i]}\n' if self.query_name else ''
        answer = f' ### {response_key}: {example[response_col][i]}' if self.response_name else ''
        text = context + question + answer
        output_texts.append(text)
      return output_texts

    return formatting_prompts_func

  def lora_configs(self):

    if self.lora == True:
      if self.q_lora == True:
        self.load_in_8bit = False
        init_lora_weights="loftq"
        loftq_config=LoftQConfig(loftq_bits=4)
        peft_config = LoraConfig(
            init_lora_weights=init_lora_weights,
            loftq_config=loftq_config,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['q_proj', 'k_proj', 'v_proj'],
        )
      else:
        peft_config = LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['q_proj', 'k_proj', 'v_proj'],
        )

    else:
      peft_config = None

    return peft_config

  def configs(self):
    peft_config = self.lora_configs()

    # Model, dataset and Tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        self.model_name,
        use_safetensors=True,
        load_in_8bit = self.load_in_8bit,
        device_map=self.device_map,
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Tokenizer
    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # Dataset
    # Load dataset
    dataset = load_dataset(self.dataset_name, split="train")
    # response_template_with_context = f" ### {self.response_name}:"
    # response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:] 
    # collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    formatter = self.get_formatter_function()

    # Training arguments and Trainer
    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=self.output_dir,
        num_train_epochs=self.num_train_epochs,
        per_device_train_batch_size=self.per_device_train_batch_size,
        gradient_accumulation_steps=self.gradient_accumulation_steps,
        optim=self.optim,
        save_steps=self.save_steps,
        logging_steps=self.logging_steps,
        learning_rate=self.learning_rate,
        weight_decay=self.weight_decay,
        fp16=self.fp16,
        bf16=self.bf16,
        max_grad_norm=self.max_grad_norm,
        max_steps=self.max_steps,
        warmup_ratio=self.warmup_ratio,
        group_by_length=self.group_by_length,
        lr_scheduler_type=self.lr_scheduler_type,
        report_to="tensorboard"
    )

    # Set supervised fine-tuning parameters
    self.trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatter,
        # data_collator=collator,
        max_seq_length=self.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=self.packing,
    )

  def start_training(self):
    self.trainer.train()

    # Save trained model
    final_adaptor_path = f'{self.model_name}_peft_adaptors'
    self.trainer.model.save_pretrained(final_adaptor_path)

    return final_adaptor_path

  def tensorboard(self):
    tensorboard_port = self.tensorboard_port
    logdir = self.logdir
    os.system(f'tensorboard -- logdir={logdir} -- port={tensorboard_port}')

  def training_and_logging(self):
    training_process = multiprocessing.Process(target=self.start_training)
    training_process.start()
    time.sleep(120)

    tensorboard_process = multiprocessing.Process(target=self.tensorboard)
    tensorboard_process.start()

    training_process.join()
    tensorboard_process.join()

  def run(self):
    self.configs()
    peft_model_path = self.start_training()
    merge_peft_adaptors(self.model_name, peft_model_path, self.fine_tuned_model_name, './model_weights')

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument('-p', '--config_path', type=str)
  args = parser.parse_args()
  my_dict_path = args.config_path
  
  llm_tr = llm_training_hf(my_dict_path)
  llm_tr.run()
