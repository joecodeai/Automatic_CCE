import os
import importlib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
import torch
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json

import prompts
import evaluate
import argparse

'''
nohup python3 -u train.py --model='mistral' --data='acl_arc' --schema='XML' >out2.txt
'''

parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--data')  #acl_arc, finecite, multicite
parser.add_argument('--schema') # XML, JSON

args = parser.parse_args()

MODEL = args.model
DATA = args.data
SCHEMA = args.schema
DATA_SIZE = None # 1000


INPUT = f'./data/{DATA}/{SCHEMA}/'
OUTPUT = f'./output/{DATA}/{MODEL}/{SCHEMA}/'
if DATA_SIZE:
    INPUT = f'./data/{DATA}/{DATA_SIZE}/{SCHEMA}/'
    OUTPUT = f'./output/{DATA}/{MODEL}/{DATA_SIZE}/{SCHEMA}/'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_mapping = {
    "mistral":'mistralai/Mistral-7B-Instruct-v0.3',
    "scitulu":'allenai/scitulu-7b',
    'llama':'meta-llama/Meta-Llama-3.1-8B-Instruct',
}
model_id = model_mapping[MODEL]

max_seq_length = 1024

#tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

#model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.bfloat16,
)

LMmodel = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = bnb_config,
    torch_dtype = torch.bfloat16,
    device_map ='auto'
)
LMmodel.resize_token_embeddings(len(tokenizer))

peft_config = LoraConfig(target_modules=[ "v_proj", "q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj" ], inference_mode=False, r=4, lora_alpha=32, lora_dropout=0.1)

LMmodel = get_peft_model(LMmodel, peft_config)

LMmodel.print_trainable_parameters()

# load data
with open(INPUT + 'train.json', 'r') as file:
    train_data = json.load(file)
    
with open(INPUT + 'test.json', 'r') as file:
    test_data = json.load(file)

# Convert the DataFrame to a Dataset
train_ds = Dataset.from_list(train_data)
test_ds = Dataset.from_list(test_data[:int(len(test_data)/2)])

# initialize prompt class

prompt = prompts.PromptForAutoCCA(tokenizer, DATA, SCHEMA)

#Apply the tokenization function to the dataset
train_ds = train_ds.map(
    lambda row: prompt.create_sample(row['input'], row['output']), 
    batched=False, 
    remove_columns=train_ds.column_names
)

dev_ds = test_ds.map(
    lambda row: prompt.create_sample(row['input'], row['output']), 
    batched=False, 
    remove_columns=test_ds.column_names
)

eval_ds = test_ds.map(
    lambda row: prompt.create_sample(row['input'], row['output'], for_generation=True), 
    batched=False, 
)

# Define Data Collator
class CustomDataCollator:
    def __init__(self, tokenizer, padding, max_length):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors='pt',
        )
        labels = batch["input_ids"].clone()
        
        # Compute loss mask for output token only
        for i in range(batch['input_ids'].shape[0]):
            
            # Decode whole sample
            text_content = self.tokenizer.decode(batch['input_ids'][i][1:])  
            
            # Extract output boundary
            output_boundary = text_content.rfind("[/INST]") + len("[/INST]")
            prompt_text = text_content[:output_boundary]
            
            # tokenize promt text
            prompt_text_tokenized = self.tokenizer(
                prompt_text,
                return_overflowing_tokens=False,
                return_length=False,
            )
            # get length of promt text
            promt_text_len = len(prompt_text_tokenized['input_ids'])
            
            # set loss mask for promt text
            labels[i][range(promt_text_len)] = -100
            
                    
        batch["labels"] = labels
        return batch

# init data collator
data_collator=CustomDataCollator(
    tokenizer=tokenizer, 
    padding="longest", 
    max_length=max_seq_length, 
)

# hyper parameterh
params =  {
    'lr': 1e-5
}

# load trainer
training_arguments = TrainingArguments(
    output_dir=OUTPUT,
    eval_strategy = 'steps',
    eval_steps= 500,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=params['lr'],
    #num_train_epochs = 9,
    max_steps=len(train_data) * 3,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps = 100,
    save_strategy = 'steps',
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    label_names = ['labels'],
)
trainer = Trainer(
    model=LMmodel,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    tokenizer=tokenizer,
    args=training_arguments,
    data_collator=data_collator,
    callbacks= [EarlyStoppingCallback(early_stopping_patience=1)]
)

# par? 2500 steps wiht 1e-04? SFTTrainer? arr as answer

print('\n### Start Training ###\n')
trainer.train()

print('\n### Start Evaluation ###\n')

evaluator = evaluate.Evaluator(trainer.model, model_id, tokenizer, eval_ds, DATA, SCHEMA, DEVICE)
results = evaluator.evaluate() #test_data=eval_ds['output'])

results['params'] = params
results['notes'] = ''

with open(OUTPUT + 'eval_results.jsonl', 'a') as fp:
    json.dump(results, fp)