import os
import pandas as pd
import numpy as np
import csv
import pickle
from transformers import Trainer, AutoModelForSequenceClassification, logging
import warnings
import datasets
from common import spacify_aa, tokenize_function_factory
from conceptual_utils import (
    Mutation,
    mutate_sequence,
    process_sklearn,
    process_huggingface,
)

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
datasets.set_caching_enabled(False)

dataset_path = str(snakemake.input["dataset"]).rsplit("/", 1)[0]
mutation_path = snakemake.input["mutations"]
model_path = snakemake.input["model"]

tokenizer = snakemake.params.get("tokenizer")
fold = int(snakemake.params["fold"])
max_length = snakemake.params["max_length"]
targets = snakemake.params["targets"]

mode = snakemake.params["mode"]

results_path = str(snakemake.output["results"]).rsplit("/", 1)[0]


if mode == "sklearn":
    process_func = process_sklearn
    tokenize_func = lambda x: x
    model = pickle.load(open(model_path, mode="rb"))
elif mode == "transformer":
    process_func = process_huggingface
    tokenize_func = tokenize_function_factory(tokenizer, max_length=max_length)
    model_path = str(model_path).rsplit("/", 1)[0]
    model = Trainer(
        model=AutoModelForSequenceClassification.from_pretrained(model_path)
    )
else:
    raise ValueError(f"Did not understand mode: {mode}")


dataset = datasets.Dataset.load_from_disk(dataset_path)

test_key = lambda example: (int(example["fold"]) == int(fold)) & (
    len(example["sequence"]) == max_length
)
test_dataset = dataset.filter(test_key)
assert len(test_dataset) > 10


if mode == "transformer":
    test_dataset = test_dataset.map(spacify_aa).map(tokenize_func, batched=True)

orig_predictions = process_func(model, test_dataset, targets)

all_predictions = {"sampled": orig_predictions}

with open(mutation_path) as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        mutation = Mutation.from_name(row["name"])
        mut_func = lambda ex: mutate_sequence(mutation, ex)
        mut_dataset = (
            test_dataset.map(mut_func, batched=True)
            .map(spacify_aa)
            .map(tokenize_func, batched=True)
        )
        all_predictions[row["name"]] = process_func(model, mut_dataset, targets)
        all_predictions[row["name"]]["change"] = mut_dataset["desc"]

d_dict = dict(
    (key, datasets.Dataset.from_pandas(df)) for key, df in all_predictions.items()
)
all_preds = datasets.DatasetDict(d_dict)
all_preds.save_to_disk(results_path)
