# ! pip install -q torch peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 accelerate einops tqdm scipy

import os
from dataclasses import dataclass, field
from typing import Optional

from peft import LoraConfig, prepare_model_for_kbit_training

import torch
from datasets import load_dataset, load_from_disk
from transformers import (
AutoModelForCausalLM,
AutoTokenizer,
BitsAndBytesConfig,
HfArgumentParser,
TrainingArguments                        
)

from tqdm.notebook import tqdm
import pandas as pd
# pip install --upgrade trl
# %pip install -U datasets

from trl import *

dataset = load_dataset("Amod/mental_health_counseling_conversations", split = "train")
df = pd.DataFrame(dataset)

def format_row(row):
  question = row['Context']
  answer = row['Response']
  fromatted_string = f"[INST] {question} [/INST] {answer}"
  return fromatted_string

df['Formatted'] = df.apply(format_row, axis = 1)
new_df = df.rename(columns = {'Formatted': 'Text'})
new_df = new_df[['Text']]

new_df.to_csv("formatted_data.csv", index=False)

training_dataset = load_dataset("csv", data_files='formatted_data.csv', split='train') 

base_model = "microsoft/phi-2"
new_model =  "MentalHealth"

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant = False
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    flash_attn=True,
    flash_rotary=True,
    fused_dense=True,
    low_cpu_mem_usage=True,
    device_map="auto",
    revision="refs/pr/23"
)

training_arguments = TrainingArguments(
    output_dir = "/kaggle/working/mhGPT_",
    num_train_epochs = 2,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 32,
    evaluation_strategy = "steps",
    eval_steps = 1500,
    logging_steps = 15,
    optim = "paged_adamw_8bit",
    learning_rate = 2e-4,
    lr_scheduler_type ="cosine",
    save_steps = 1500,
    warmup_ratio = 0.05,
    weight_decay=0.01,
    max_steps=-1,
)

model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

peft_config = LoraConfig(
    r=64,
    lora_alpha = 64,
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = ["Wqkv", "fc1", "fc2"]
)


trainer = SFTTrainer(
    model = model,
    train_dataset = training_dataset,
    peft_config = peft_config,
    dataset_text_field = "Text",
    max_seq_length = 400,
    tokenizer = tokenizer,
    args = training_arguments
)


trainer.train()

trainer.model.save_pretrained("mentalhealthgpt_v1.2")




