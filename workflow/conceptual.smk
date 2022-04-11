rule conceptual_all:
    input:
        expand('concept/{model}_{stem}/{fold}/dataset_dict.json',
               model=MODELS, fold=FOLDS,
               stem=['V3_coreceptor', 'PR_resist'])


def conceptual_mode(wildcards):
    if 'bert' in wildcards['model']:
        return 'transformer'
    else:
        return 'sklearn'


rule conceptual_error:
    shadow: "shallow"
    resources:
        gpu = gpu_size,
        mem = 10,
    script:
        'scripts/conceptual_error.py'


use rule conceptual_error as PR_conceptual with:
    input:
        dataset = 'datasets/PR_resist/dataset_info.json',
        mutations = 'data/PR_muts.tsv',
        model = 'models/{model}_PR_resist/{fold}/config.json'
    output:
        results = 'concept/{model}_PR_resist/{fold}/dataset_dict.json'
    params:
        fold = lambda wildcards: wildcards['fold'],
        tokenizer = TOKENIZER,
        mode = conceptual_mode,
        max_length = 99,
        targets = ['FPV', 'IDV', 'NFV', 'SQV']

use rule conceptual_error as Corecep_conceptual with:
    input:
        dataset = 'datasets/V3_coreceptor/dataset_info.json',
        mutations = 'data/Co_muts.tsv',
        model = 'models/{model}_V3_coreceptor/{fold}/config.json'
    output:
        results = 'concept/{model}_V3_coreceptor/{fold}/dataset_dict.json'
    params:
        fold = lambda wildcards: wildcards['fold'],
        tokenizer = TOKENIZER,
        mode = conceptual_mode,
        max_length = 35,
        targets = ['CCR5', 'CXCR4'],
