import os

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS']='1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import datasets
import re
import numpy as np
from common import spacify_aa, tokenize_function_factory
from itertools import islice

from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling


dataset_path = str(snakemake.input['dataset']).rsplit('/',1)[0]
out_model = str(snakemake.output['model']).rsplit('/',1)[0]

print(dataset_path, out_model)

pretrained = snakemake.input.get('pretrained', None)
if pretrained is None:
    pretrained = snakemake.params.get('pretrained', None)

assert pretrained is not None, 'pretrained must be specified in either input, params, or in the model meta'    
    
trainer_path = str(snakemake.output['trainer'])

# Grab parameters for training
tokenizer = snakemake.params['tokenizer']
max_length = snakemake.params.get('max_length', 128)
targets = snakemake.params['targets']
mlm_prob = snakemake.params.get('mlm_prob', 0.15)

EPOCHS = snakemake.params.get('epochs', 200)

callbacks = []
if snakemake.params.get('early_stopping', True):
    callbacks.append(EarlyStoppingCallback(
        early_stopping_patience=snakemake.params.get('early_stopping', 3)))



def flatten_prots(examples):
    for p in targets:
        for prot in examples[p]:
            for l in prot:
                yield l

def chunkify(it, max_size):
    items = list(islice(it, max_size))
    while items:
        yield items
        items = list(islice(it, max_size))
        
        
def chunk_proteins(examples):
    chunks = chunkify(flatten_prots(examples), max_length)
    return {'sequence': [''.join(c) for c in chunks]}


train_key = lambda example: example['fold'] != fold
test_key = lambda example: example['fold'] == fold

if type(tokenizer) is str:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, do_lower_case=False)
    
dataset = datasets.Dataset.load_from_disk(dataset_path)

chunked_set = dataset.map(chunk_proteins, 
                          remove_columns=dataset.column_names, 
                          batched=True)

tkn_func = tokenize_function_factory(tokenizer = tokenizer, 
                                     max_length = max_length)

tokenized_dataset = chunked_set.map(spacify_aa).map(tkn_func, batched=True)
split_dataset = tokenized_dataset.train_test_split()

def model_init():
    return AutoModelForMaskedLM.from_pretrained(pretrained)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=mlm_prob,
    pad_to_multiple_of=8,
)

training_args = TrainingArguments(trainer_path,
                                  evaluation_strategy='epoch',
                                  load_best_model_at_end=True,
                                  save_strategy='epoch',
                                  logging_first_step=True,
                                  logging_steps=10,
                                  num_train_epochs=EPOCHS,
                                  warmup_steps=50,
                                  weight_decay=0.01,
                                  per_device_train_batch_size = 16,
                                  per_device_eval_batch_size = 16,
                                  gradient_accumulation_steps=64,
                                  lr_scheduler_type='cosine_with_restarts'
                                  )

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=split_dataset['train'],
    eval_dataset=split_dataset['test'],
    data_collator=data_collator,
    callbacks = callbacks
)

results = trainer.train()

trainer.save_model(out_model)