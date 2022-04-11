import datasets
import re
import numpy as np

dataset_path = str(snakemake.input).rsplit('/',1)[0]
outdir = str(snakemake.output).rsplit('/',1)[0]

tokenizer = snakemake.params['tokenizer']
fold = int(snakemake.params['fold'])
max_length = snakemake.params['max_length']
targets = snakemake.params['targets']

dataset = datasets.Dataset.load_from_disk(dataset_path)

train_key = lambda example: example['fold'] != fold
test_key = lambda example: example['fold'] == fold

if type(tokenizer) is str:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, do_lower_case=False)


def spacify_aa(example):
    """Do the AA space thing and mask rare AAs (as suggested in the docs)"""
    spaced_aa = ' '.join(list(re.sub(r"[UZOB]", "X", example['sequence'])))
    return {'sentence': spaced_aa}    
    
    
def tokenize_function(examples):
    return tokenizer(examples["sentence"],
                     padding="max_length",
                     max_length = max_length,
                     truncation=True)


tokenized_dataset = dataset.map(spacify_aa).map(tokenize_function, batched=True)

def create_multilabels(ex):
    return {'labels': np.array([ex[t] for t in targets]).reshape(1,-1)}

if len(targets) > 1:
    labeled_dataset = tokenized_dataset.map(create_multilabels)
else:
    labeled_dataset = tokenized_dataset.rename_column(targets[0], 'labels')

split_dset = {'train': labeled_dataset.filter(train_key),
              'test': labeled_dataset.filter(test_key)}

split_dataset = datasets.DatasetDict(split_dset)

split_dataset.save_to_disk(outdir)