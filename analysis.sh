#!/bin/bash
#SBATCH -p gpu24,gpu22,gpu17
#SBATCH -c 12
#SBATCH --gres gpu:1 
#SBATCH --mem 90G
#SBATCH --exclude gpu22-a40-[01-12],gpu17-l40-20
#SBATCH -t 0-10
#SBATCH -o ./%A.out
#SBATCH -e ./%A.err

CONDA=/BS/mparcham2/work/miniforge3/bin/conda
CONDA_ENV_NAME=OtherPie


ARCH='densenet121_long'; PAIRS=(
  # FaCT models
  "features.norm5,16384,16,0.001,BiasFreeTopK"
);

for ENTRY in "${PAIRS[@]}"; do
    IFS=',' read -r LAYER K ALPHA LR IMPL<<< "$ENTRY"

    EXPNAME="${IMPL}";
    echo "${LAYER}: ${EXPNAME}-${K}-${ALPHA}-LR${LR}"

    ##### Accuracy, activation, and contribution statistics
    ## Required for other analysis scripts!
    ${CONDA} run -n ${CONDA_ENV_NAME} --no-capture-output \
      python3 -um fact.analysis.analysis \
        --exp_name ${EXPNAME} --method ${IMPL} \
        --layer ${LAYER} --arch ${ARCH} --nr_concepts ${K} --sparsity ${ALPHA} \
        --lr ${LR} --bcos_layers --download v0.1
    
    ##### Concept Consistency
    # ${CONDA} run -n ${CONDA_ENV_NAME} --no-capture-output \
    #   python3 -um fact.analysis.dino_consistency \
    #     --method ${IMPL} --exp_name ${EXPNAME} \
    #     --layer ${LAYER} --arch ${ARCH} --nr_concepts ${K} --sparsity ${ALPHA} \
    #     --lr ${LR} --bcos_layers --download v0.1 --quantile_per_concept 0.95 # i.e Top 5%
    
    ##### Visualize Concepts
    ${CONDA} run -n ${CONDA_ENV_NAME} --no-capture-output \
      python3 -um fact.plotting.plot_inference \
        --method ${IMPL} --exp_name ${EXPNAME} \
        --layer ${LAYER} --arch ${ARCH} --nr_concepts ${K} --sparsity ${ALPHA} \
        --lr ${LR} --bcos_layers --download v0.1
done

# ARCH='densenet121_long'; PAIRS=(
  ## FaCT models
  # "features.norm5,16384,16,0.001,BiasFreeTopK"
  # "features.transition3,16384,32,0.001,BiasFreeTopK"
  # "features.transition2,8192,32,0.0001,BiasFreeTopK"
  ## Channels of B-cos (Remove --download for these)
  # "features.norm5,0,0,0,channels"
  # "features.transition3,0,0,0,channels"
  # "features.transition2,0,0,0,channels"
# );

# ARCH='resnet50_long'; PAIRS=(
  ## FaCT models
  # "layer4,16384,16,0.001,BiasFreeTopK" 
  # "layer3,16384,32,0.001,BiasFreeTopK"
  # "layer2,8192,32,0.001,BiasFreeTopK"
  ## Channels of B-cos (Remove --download for these)
  # "layer4,0,0,0,channels,"
  # "layer3,0,0,0,channels"
  # "layer2,0,0,0,channels"
# );

# ARCH='vitc_s_patch1_14'; PAIRS=(
  ## FaCT models
  # "0.transformer.encoder_10,8192,32,0.001,BiasFreeTopK"
  # "0.transformer.encoder_9,16384,32,0.0001,BiasFreeTopK"
  # "0.transformer.encoder_8,16384,32,0.001,BiasFreeTopK"
  # "0.transformer.encoder_4,16384,64,0.0001,BiasFreeTopK"
  ## Channels of B-cos (Remove --download for these)
  # "0.transformer.encoder_10,0,0,0,channels"
  # "0.transformer.encoder_9,0,0,0,channels"
  # "0.transformer.encoder_8,0,0,0,channels"
  # "0.transformer.encoder_4,0,0,0,channels"
# );

## For evaluating the DiNo Consistency of standard models, you can the following configs.
## Make sure to remove the --bcos_layer and --download from the args
## and run the `analysis` script beforehand for the CRP method.
# ARCH='std_resnet50'; PAIRS=(
#   "layer4,0,0,0,crp"
# );
# ARCH='std_resnet50'; PAIRS=(
#   "layer4,0,0,0,craft"
# );