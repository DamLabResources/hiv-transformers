# Wrapper for training HuggingFace Models

Refines a huggingface model against a prediction task.
It takes an input Huggingface `DatasetDict` (with a `train` and `test` dataset split) that has (at least) the following fields:
  - `labels` : A `ClassLabel` (For single-class predictions), `Array2D` (for multi-class prediction), anything else (for single regression tasks)
  - `attention_mask` : Tokenized values
  - `input_ids` : Tokenized values
  - `token_type_ids` : Tokenized values


## Example:

```
rule train:
  input:
    dataset = 'path/to/tokenized/dataset',
  output:
    model = 'path/to/refined/model'
    metric = 'path/to/refined/model/metric'
  params:
    pretrained = 'protbert', #Can also be path or used as input
    epoch = 200, # default
    early_stopping = True, # default
    
    



```
