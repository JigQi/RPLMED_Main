#!/bin/bash

# custom config
DATA=$1
DATASET=$2

NCTX=4
CSC=False
CTP=end

METHOD=RPLMED
TRAINER=RPLMED_BiomedCLIP

SHOTS_LIST=(1 2 4 8 16)

echo "Running few-shot experiments for shots: ${SHOTS_LIST[@]}"

for SHOTS in ${SHOTS_LIST[@]}; do
    for SEED in 1 2 3; do
        DIR=output/${DATASET}/shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            echo "Running: SHOTS=${SHOTS}, SEED=${SEED}"
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml  \
            --output-dir ${DIR} \
            TRAINER.RPLMED.N_CTX ${NCTX} \
            TRAINER.RPLMED.CSC ${CSC} \
            TRAINER.RPLMED.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS}
        fi
    done
done