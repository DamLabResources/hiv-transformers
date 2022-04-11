import pandas as pd
import datasets
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import numpy as np
from yaml import dump
import pickle


from common import multiclass_metrics_factory, PredictionMock

# Prep the inputs
dataset_path = str(snakemake.input['dataset']).rsplit('/',1)[0]
labels = snakemake.params.get('labels')

dummy_metric_output = snakemake.output['dummy_metric']
tfid_metric_output = snakemake.output['tfid_metric']

dummy_model_output = snakemake.output['dummy_model']
tfid_model_output = snakemake.output['tfid_model']

# Prep the labels
id2label = dict((n, label) for n, label in enumerate(labels))
label2id = dict((label, n) for n, label in enumerate(labels))
num_labels = len(labels)

metric_func = multiclass_metrics_factory(id2label, expect_logits = False)


# Get the data

dset = datasets.DatasetDict.load_from_disk(dataset_path)

downsample = snakemake.params.get('downsample', None)
if downsample is None:
    Xtrain = np.array(dset['train']['sequence'])
    ytrain = np.array(dset['train']['labels']).reshape(-1, num_labels)
else:
    frac = downsample/len(dset['train'])
    ds_dataset = dset['train'].filter(lambda x: np.random.rand() < frac)

    Xtrain = np.array(ds_dataset['sequence'])
    ytrain = np.array(ds_dataset['labels']).reshape(-1, num_labels)



Xtest = np.array(dset['test']['sequence'])
ytest = np.array(dset['test']['labels']).reshape(-1, num_labels)


# Build the models

dummy_model = DummyClassifier(strategy = 'stratified')

tfid_model = make_pipeline(TfidfVectorizer(analyzer = 'char', 
                                           ngram_range=(1,3)),
                                VarianceThreshold(),
                                RandomForestClassifier())


# Train the models
dummy_model.fit(Xtrain, ytrain)
tfid_model.fit(Xtrain, ytrain)

# Predict held-out data
tfid_preds = tfid_model.predict_proba(Xtest)
dummy_preds = dummy_model.predict_proba(Xtest)

# Get metrics
tfid_results = metric_func(PredictionMock(tfid_preds, ytest))
dummy_results = metric_func(PredictionMock(dummy_preds, ytest))

for key in tfid_results.keys():
    tfid_results[key] = float(tfid_results[key])
    dummy_results[key] = float(dummy_results[key])

with open(dummy_metric_output, 'w') as handle:
    dump(dummy_results, handle)

pickle.dump(dummy_model, open(dummy_model_output, "wb"))
    
    
with open(tfid_metric_output, 'w') as handle:
    dump(tfid_results, handle)
    
pickle.dump(tfid_model, open(tfid_model_output, "wb"))