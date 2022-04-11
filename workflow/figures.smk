rule dataset_figures:
    input:
        datasets = expand('datasets/{root}/dataset_info.json',
                          root=['FLT_genome', 'PR_resist',
                                'V3_coreceptor', 'V3_bodysite'])
    output:
        dataset_description = 'figures/Fig2-dataset_description-high.png'
    notebook:
        '../figures/dataset_figure.py.ipynb'

rule model_comparison_figures:
    input:
        model_results = expand('models/{stem}/{fold}/best.yaml',
                               stem=dataset_stems, fold=FOLDS)
    output:
        model_group = 'Fig3-model_group-high.png',
        model_results = 'Fig4-model_results-high.png',
        model_table = 'Table1-model_results.xlsx'
    notebook:
        '../figures/model_comparison.py.ipynb'

rule masked_learning_figures:
    input:
        dataset = 'datasets/V3_coreceptor/dataset_info.json',
        hivbert = 'models/hivbert_genome/config.json'
    output:
        masked_results = 'figures/Fig5-masked_results-high.png'
    resources:
        gpu = 10
    notebook:
        '../figures/masked_learning.ipynb'

rule mutation_scatterplot_figures:
    input:
        mutation_results = expand('concept/{model}_{stem}/{fold}/dataset_dict.json',
                                  model=MODELS, fold=FOLDS,
                                  stem=['V3_coreceptor', 'PR_resist'])
    output:
        corecept_scatter = 'Fig6-corecept_scatter-high.png',
        conceptual_model_results = 'Fig7-conceptual_model_results-high.png',
        conceptual_field_results = 'Fig8-conceptual_field_results-high.png',
    notebook:
        '../figures/mutation_scatterplot.py.ipynb'


rule figures_all:
    input:
        rules.dataset_figures.output,
        rules.model_comparison_figures.output,
        rules.masked_learning_figures.output,
        rules.mutation_scatterplot_figures.output