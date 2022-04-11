rule downsample_all:
    input:
        expand('downsample/{model}_{stem}/{fold}/{DS}-best.yaml',
               model=MODELS, stem=dataset_stems, fold=FOLDS,
               DS=[25, 50, 100, 500, 1000, 2000]),
        expand('downsample/{model}_{stem}/{fold}/{DS}/concept/dataset_dict.json',
               model=MODELS, fold=FOLDS,
               stem=['PR_resist', 'V3_coreceptor'],
               DS=[25, 50, 100, 500, 1000, 2000])


use rule huggingface_classification_training as hivbert_downsample with:
    input:
        dataset = 'splits/{prot}_{target}/{fold}/dataset_dict.json',
        pretrained = 'models/hivbert_genome/config.json'
    output:
        model = 'downsample/hivbert_{prot}_{target}/{fold}/{DS}/config.json',
        metric = 'downsample/hivbert_{prot}_{target}/{fold}/{DS}-best.yaml',
    params:
        epochs = 200,
        early_stopping = 3,
        tokenizer = TOKENIZER,
        labels = targets_func,
        downsample = lambda wildcards: int(wildcards['DS'])

use rule huggingface_classification_training as protbert_downsample with:
    input:
        dataset = 'splits/{prot}_{target}/{fold}/dataset_dict.json',
    output:
        model = 'downsample/protbert_{prot}_{target}/{fold}/{DS}/config.json',
        metric = 'downsample/protbert_{prot}_{target}/{fold}/{DS}-best.yaml',
    params:
        epochs = 200,
        early_stopping = 3,
        tokenizer = TOKENIZER,
        labels = targets_func,
        pretrained = TOKENIZER,
        downsample = lambda wildcards: int(wildcards['DS']),

use rule sklearn_train as sklearn_downsample with:
    input:
        dataset = 'splits/{prot}_{target}/{fold}/dataset_dict.json'
    output:
        dummy_metric = 'downsample/dummy_{prot}_{target}/{fold}/{DS}-best.yaml',
        dummy_model = 'downsample/dummy_{prot}_{target}/{fold}/{DS}/config.json',
        tfid_metric = 'downsample/tfid_{prot}_{target}/{fold}/{DS}-best.yaml',
        tfid_model = 'downsample/tfid_{prot}_{target}/{fold}/{DS}/config.json',
    params:
        labels = targets_func,
        downsample = lambda wildcards: int(wildcards['DS']),

use rule PR_conceptual as PR_downsampled_conceptual with:
    input:
        dataset = 'datasets/PR_resist/dataset_info.json',
        mutations = 'data/PR_muts.tsv',
        model = 'downsample/{model}_PR_resist/{fold}/{DS}/config.json'
    output:
        results = 'downsample/{model}_PR_resist/{fold}/{DS}/concept/dataset_dict.json'
    params:
        fold = lambda wildcards: wildcards['fold'],
        tokenizer = TOKENIZER,
        mode = conceptual_mode,
        max_length = 99,
        targets = ['FPV', 'IDV', 'NFV', 'SQV']

use rule Corecep_conceptual as Corecep_downsampled_conceptual with:
    input:
        dataset = 'datasets/V3_coreceptor/dataset_info.json',
        mutations = 'data/Co_muts.tsv',
        model = 'downsample/{model}_V3_coreceptor/{fold}/{DS}/config.json'
    output:
        results = 'downsample/{model}_V3_coreceptor/{fold}/{DS}/concept/dataset_dict.json'
    params:
        fold = lambda wildcards: wildcards['fold'],
        tokenizer = TOKENIZER,
        mode = conceptual_mode,
        max_length = 35,
        targets = ['CCR5', 'CXCR4'],