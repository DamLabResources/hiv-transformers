MODEL_ORDER = ["dummy", "tfid", "protbert", "hivbert"]
MODEL_FANCY_NAMES = {"dummy": 'Null-Model',
                     "tfid": 'TF-IDF',
                     "protbert": 'Prot-BERT',
                     "hivbert": 'HIV-BERT'}
MODEL_FANCY_ORDER = [MODEL_FANCY_NAMES[t] for t in MODEL_ORDER]

TASK_ORDER = ["resist", "coreceptor", "bodysite"]
TASK_FANCY_NAMES = {"resist": 'Protease\nResistance', 
                    "coreceptor": 'Coreceptor',
                    "bodysite": 'Tissue'}
TASK_FANCY_ORDER = [TASK_FANCY_NAMES[t] for t in TASK_ORDER]

DRUG_ORDER = ['FPV', 'IDV', 'NFV', 'SQV']
DRUG_CMAP = 'Set3'

CORECEPTOR_ORDER = ['CCR5', 'CXCR4']
CORECEPTOR_CMAP = 'Dark2'

CORECEPTOR_EFFECT_MUTS = ['E25H', 'E25K', 'E25R',
                          'G24H', 'G24K', 'G24R',
                          'S11H', 'S11K', 'S11R']

CORECEPTOR_NEUTRAL_MUTS = ['C35A', 'D29E', 'G17A',
                           'H13Y', 'H34F', 'N6M',
                           'P4N', 'R9E', 'S11R']

RESIST_EFFECT_MUTS = ['D30N', 'V32I', 'M46I',
                      'M46L', 'G48V', 'I54V',
                      'V82F', 'I84C', 'N88S',
                      'L90M']
RESIST_NEUTRAL_MUTS = ['D60N', 'V56I', 'M36I',
                       'M36L', 'G31V', 'I13V',
                       'V77F', 'I66C', 'N37S',
                       'L10M']


TISSUE_ORDER = [
        "periphery-tcell",
        "periphery-monocyte",
        "CNS",
        "breast-milk",
        "female-genitals",
        "male-genitals",
        "gastric",
        "lung",
        "organ",
    ]
TISSUE_CMAP = 'Paired'


HISTOGRAM_CMAP = "Greens"
COUNTING_CMAP = "Set2"
MODEL_CMAP = 'Set1'


TOP_CORNER_FIGSIZE = (4, 4)
FULL_WIDTH_FIGSIZE = (7, 4)
HALF_PAGE_FIGSIZE = (7, 6)
FULL_PAGE_FIGSIZE = (7, 10)