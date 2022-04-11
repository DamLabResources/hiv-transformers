MODELS = ['protbert',
          'hivbert',
          'dummy',
          'tfid']

FOLDS = range(5)


def targets_func(wildcards):
    "Get targets of the function based on the {target} wildcard."

    if 'coreceptor' in wildcards['target']:
        return ['CCR5', 'CXCR4']
    elif 'resist' in wildcards['target']:
        return ['FPV', 'IDV', 'NFV', 'SQV']
    elif 'bodysite' in wildcards['target']:
        return ['periphery-tcell', 'periphery-monocyte',
                'CNS', 'breast-milk',
                'female-genitals', 'male-genitals',
                'gastric', 'lung',  'organ']
    else:
        raise ValueError(f'Did not understand {wildcards["target"]}')


def model_max_length(wildcards):
    "Return a biologically relevant maximum length for model padding based on the {prot} wildcard"

    try:
        if 'V3' in wildcards['prot']:
            return 40
        elif 'PR' in wildcards['prot']:
            return 110
        else:
            raise ValueError(f'Did not understand {wildcards["prot"]}')
    except KeyError:
        raise KeyError('Missing the {prot} wildcard to use this function.')


def gpu_size(wildcards, attempt):
    """Return model size in gb based on {prot} and {model}"""

    prot = wildcards.get('prot', 'V3')
    model = wildcards.get('model', 'protbert')

    if 'bert' not in model:
        return 0
    if prot == 'V3':
        return 9*attempt
    return 12*attempt
