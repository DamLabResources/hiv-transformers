rule train_all:
    input:
        expand('models/{model}_{stem}/{fold}/best.yaml',
               stem=dataset_stems,
               fold=FOLDS,
               model=MODELS),

rule genome_hivbert_train:
    input:
        dataset = 'datasets/FLT_genome/dataset_info.json',
    output:
        model = 'models/hivbert_genome/config.json',
        trainer = directory('scratch/trainers/hivbert/'),
    params:
        pretrained = 'Rostlab/prot_bert_bfd',
        tokenizer = TOKENIZER,
        targets = ['GagPol', 'Vif', 'Vpr', 'Tat', 'Rev', 'Vpu', 'Env', 'Nef']
    resources:
        gpu = 20,
        mem = 10,
    script:
        'scripts/huggingface_lm.py'


rule huggingface_classification_training:
    resources:
        gpu = gpu_size,
        mem = 10,
    shadow: "shallow"
    params:
        epochs = 200,
        early_stopping = 3,
        tokenizer = TOKENIZER,
        labels = targets_func,
    script:
        'scripts/huggingface_train.py'


use rule huggingface_classification_training as hivbert_training with:
    input:
        dataset = 'splits/{prot}_{target}/{fold}/dataset_dict.json',
        pretrained = 'models/hivbert_genome/config.json'
    output:
        model = 'models/hivbert_{prot}_{target}/{fold}/config.json',
        metric = 'models/hivbert_{prot}_{target}/{fold}/best.yaml',
    params:
        epochs = 200,
        early_stopping = 3,
        tokenizer = TOKENIZER,
        labels = targets_func,

use rule huggingface_classification_training as protbert_training with:
    input:
        dataset = 'splits/{prot}_{target}/{fold}/dataset_dict.json'
    output:
        model = 'models/protbert_{prot}_{target}/{fold}/config.json',
        metric = 'models/protbert_{prot}_{target}/{fold}/best.yaml',
    params:
        epochs = 200,
        early_stopping = 3,
        tokenizer = TOKENIZER,
        labels = targets_func,
        pretrained = TOKENIZER


rule sklearn_train:
    input:
        dataset = 'splits/{prot}_{target}/{fold}/dataset_dict.json'
    output:
        dummy_metric = 'models/dummy_{prot}_{target}/{fold}/best.yaml',
        dummy_model = 'models/dummy_{prot}_{target}/{fold}/config.json',
        tfid_metric = 'models/tfid_{prot}_{target}/{fold}/best.yaml',
        tfid_model = 'models/tfid_{prot}_{target}/{fold}/config.json',
    resources:
        mem = 20,
    params:
        labels = targets_func,
    script:
        'scripts/sklearn_train.py'


rule tokenize_fold:
    input:
        'datasets/{prot}_{target}/dataset_info.json'
    output:
        'splits/{prot}_{target}/{fold}/dataset_dict.json'
    params:
        fold = lambda wildcards: wildcards['fold'],
        tokenizer = TOKENIZER,
        max_length = model_max_length,
        targets = targets_func
    script:
        'scripts/tokenize_fold.py'