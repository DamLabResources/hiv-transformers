from yaml import full_load, dump
from datasets import load_dataset, DatasetDict, ClassLabel, Array2D, Sequence
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, mean_absolute_error, max_error, r2_score
from sklearn.preprocessing import label_binarize
from transformers import TrainingArguments, logging
from transformers import EarlyStoppingCallback, AutoModelForSequenceClassification
import torch
import os
from scipy.special import softmax
from collections import Counter
import numpy as np
import pandas as pd

from common import CustomTrainer, categorical_metrics_factory, multiclass_metrics_factory, regression_metrics_factory

import warnings
warnings.filterwarnings("ignore")

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS']='1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


dataset_path = str(snakemake.input['dataset']).rsplit('/',1)[0]


# Get the model and init function
pretrained = snakemake.input.get('pretrained', None)
if pretrained is None:
    pretrained = snakemake.params.get('pretrained', None)

assert pretrained is not None, 'pretrained must be specified in either input, params, or in the model meta'

if pretrained.endswith('.json'):
    pretrained = pretrained.rsplit('/',1)[0]

dataset = DatasetDict.load_from_disk(dataset_path)
label_type = type(dataset['train'].features['labels'])

if label_type == ClassLabel:
    # Classification task

    multi_class = False

    id2label = dict((n, label) for n, label in enumerate(
        dataset['train'].features['labels'].names))
    label2id = dict((label, n) for n, label in enumerate(
        dataset['train'].features['labels'].names))
    num_labels = dataset['train'].features['labels'].num_classes

    labels = pd.Series(dataset['train']['labels'])
    label_counts = labels.value_counts().reindex(range(num_labels))
    train_weights = (len(labels)/label_counts.values).astype(np.float32)

    typ = 'categorical'
    metrics = categorical_metrics_factory(id2label)

elif (label_type == Array2D) | (label_type == Sequence):

    multi_class = True

    labels = snakemake.params.get('labels')
    num_labels = len(labels) #dataset['train'].features['labels'].shape[1]
    
    if labels is None:
        labels = [f'CLASS_{i}' for i in num_labels]
    
    id2label = dict((num, cl) for num, cl in enumerate(labels))
    label2id = dict((val, key) for key, val in id2label.items())

    labels = np.array(dataset['train']['labels'])
    train_weights = (labels == 1).sum()/(labels == 0).sum()

    typ = 'multi_class'
    metrics = multiclass_metrics_factory(id2label)

else:
    # Regression task
    name = 'raw_value'
    id2label = {0: name}
    label2id = {name: 0}
    num_labels = 1
    train_weights = None

    typ = 'regression'
    metrics = regression_metrics_factory(id2label)

    
    
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(pretrained,
                                                              num_labels=num_labels,
                                                              label2id=label2id,
                                                              id2label=id2label)


# Grab parameters for training
EPOCHS = snakemake.params.get('epochs', 200)

downsample = snakemake.params.get('downsample', None)

callbacks = []
if snakemake.params.get('early_stopping', True):
    callbacks.append(EarlyStoppingCallback(
        early_stopping_patience=snakemake.params.get('early_stopping', 3)))

# Best trained model
out_model = snakemake.output.get('model', None)
out_model = str(out_model).rsplit('/',1)[0]

trainer_path = str(snakemake.output.get('trainer', 'test_trainer'))


# Metric
out_metric = snakemake.output['metric']

keep_cols = {'labels', 'attention_mask', 'input_ids', 'token_type_ids'}
rm_cols = [col for col in dataset.column_names['train']
           if col not in keep_cols]

trim_dataset = dataset.remove_columns(rm_cols)

# Based on Rostlab github example.
training_args = TrainingArguments(trainer_path,
                                  evaluation_strategy='epoch',
                                  load_best_model_at_end=True,
                                  save_strategy='epoch',
                                  logging_first_step=True,
                                  logging_steps=10,
                                  num_train_epochs=EPOCHS,
                                  warmup_steps=50,
                                  weight_decay=0.01,
                                  gradient_accumulation_steps=64,
                                  lr_scheduler_type='cosine_with_restarts',
                                  )

if downsample is None:
    train_dataset = trim_dataset['train']
else:
    frac = downsample/len(trim_dataset['train'])
    train_dataset = trim_dataset['train'].filter(lambda x: np.random.rand() < frac)


trainer = CustomTrainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=trim_dataset['test'],
    callbacks=callbacks,
    compute_metrics=metrics,
    class_weights=train_weights,
    train_type=typ
)

results = trainer.train()

if out_model is not None:
    trainer.save_model(out_model)

test_predictions = trainer.predict(trim_dataset['test'])

with open(out_metric, 'w') as handle:
    dump(test_predictions.metrics, handle)