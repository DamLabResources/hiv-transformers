DATASET_FILES = ['dataset_info.json',
                 'dataset.arrow',
                 'state.json']

rule prep_all:
    input:
        expand('datasets/{dataset}/{path}',
               path=DATASET_FILES,
               dataset=['V3_coreceptor',
                        'V3_bodysite',
                        'PR_resist',
                        'FLT_genome'])

rule prep_V3_data:
    input:
        meta_data = 'data/LANL_V3.meta.tsv',
        sequence_data = 'data/LANL_V3.fasta',
    output:
        datasets = expand('datasets/{dataset}/{path}',
                          path=DATASET_FILES,
                          dataset=['V3_coreceptor',
                                   'V3_bodysite'])
    params:
        should_upload = False
    notebook:
        'scripts/process_lanl_v3.py.ipynb'

rule prep_PR_data:
    input:
        meta_data = 'data/PI_DataSet.tsv'
    output:
        datasets = expand('datasets/{dataset}/{path}',
                          path=DATASET_FILES,
                          dataset=['PR_resist'])
    params:
        should_upload = False
    notebook:
        'scripts/process_stanford_pr.py.ipynb'

SCRATCH_PARTS = ['%03i' % i for i in range(1, 1000)]
SCRATCH_PARTS += ['%i' % i for i in range(1000, 1871)]

DNA_PART_PREFIX = 'scratch/full_genomes/HIV1_FLT_2016_genome_DNA.part_'


rule prep_FLT_data:
    input:
        transcript_files = expand(DNA_PART_PREFIX+'{num}.spliced',
                                  num=SCRATCH_PARTS)
    output:
        datasets = expand('datasets/{dataset}/{path}',
                          path=DATASET_FILES,
                          dataset=['FLT_genome'])
    params:
        should_upload = False
    notebook:
        'scripts/process_lanl_flt.py.ipynb'


rule split_HIV_FLT:
    input:
        'data/HIV1_FLT_2016_genome_DNA.fasta'
    output:
        expand(DNA_PART_PREFIX+'{num}.fasta',
               num=SCRATCH_PARTS)
    params:
        scratch = "scratch/full_genomes/"
    shell:
        "seqkit split -O {params.scratch} --by-size 1 {input}"

rule extract_genes:
    input:
        genome = DNA_PART_PREFIX+'{num}.fasta',
    output:
        DNA_PART_PREFIX+'{num}.spliced'
    threads:  # Avoiding DDoSing LANL
        lambda w: 0.25*workflow.cores
    shell:
        """
        curl -s \
        --form "seq_upload=@{input.genome}" \
        --form "region=all" \
        --form "return_format=fasta" \
        --form "translate=NO" \
        --output {output} \
        https://www.hiv.lanl.gov/cgi-bin/GENE_CUTTER/simpleGC
        """