#!/bin/bash

CONDA=/BS/mparcham2/work/miniforge3/bin/conda
CONDA_ENV_NAME=OtherPie

ARCH="densenet121_long"; LAYERS=(
	"features.norm5" # Unpooled penultimate
	# "features.transition3" #
	# "features.transition2" #
)
# ARCH="resnet50_long"; LAYERS=(
# 	layer4   # Unpooled penultimate
# 	layer3
# 	layer2
# );
# ARCH="vitc_s_patch1_14"; LAYERS=(
	# 0.transformer.encoder_5
	# 0.transformer.encoder_10
	# 0.transformer.encoder_9
	# 0.transformer.encoder_8
	# 0.transformer.encoder_4
# );

KS=(
	1024
)
LRS=(
	0.001
)
ALPHAS=(
	8
)
IMPL='BiasFreeTopK' # The training script is intended for pur bias-free topksae only
for LR in ${LRS[@]}; do
	for K in ${KS[@]}; do
		for LAYER in ${LAYERS[@]}; do
			for ALPHA in ${ALPHAS[@]}; do
				EXPNAME="${IMPL}";
				JOBNAME="${LAYER}_${EXPNAME}_K${K}_A${ALPHA}"
				CMD="${CONDA} run -n ${CONDA_ENV_NAME} --no-capture-output \
					python3 -um fact.training.train \
						--exp_name ${EXPNAME} --method ${IMPL} --bcos_layers \
						--layer ${LAYER} --arch ${ARCH} \
						--nr_concepts ${K} --sparsity ${ALPHA} --lr ${LR} \
						--max_iter 16 --batch_size 32786 \
						--num_batches 5000 \
						--precomputed_feats /scratch/inf0/user/mparcham/LargeFeats-${arch}-${layer}-N1231167.pth
				"
				# Use the above once to generate the feautres (it will exit afterwards).
				# then run again with the arg below added, to read from the stored features
				# --precomputed_feats /scratch/inf0/user/mparcham/LargeFeats-${ARCH}-L${LAYER}-N1231167.pth \

				echo "Command: [$CMD]" | tr -s '[:space:]' ' '

				# Use this for running locally
				eval "${CMD}"

				# Use this for submitting a Slurm Job
				# sbatch --job-name="${JOBNAME}" \
				# 	-o "./${JOBNAME}-%A.out" -e "./${JOBNAME}-%A.err" \
				# 	--gres gpu:1 -c 10 -p gpu24,gpu22,gpu17 \
				# 	--exclude gpu22-a40-[01-12] --mem 180G \
				#   -t 0-3:59 \
				# 	--wrap="${CMD}"
			done
		done
	done
done