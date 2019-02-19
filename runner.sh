#!/usr/bin/env bash

source ./env.sh

PYTHON=venv/bin/python

JOB_NAME=prediction_${DATE}

if [[ ${1} == "cloud" ]]; then
    export EXECUTOR=cloud
else
    export EXECUTOR=local
fi

# Switch between main actions: (pre)process, tune, train, predict
# Process data (starts Beam pipeline).
if [[ ${1} == "process" ]] | [[ ${1} == "preprocess" ]]; then

    ${PYTHON} -m src.preprocess

# Tune hyperparameters
elif [[ ${1} == "tune" ]]; then

    gcloud ml-engine jobs submit training ${JOB_NAME} \
        --stream-logs \
        --scale-tier=${SCALE_TIER} \
        --config=${HPTUNING_CONFIG} \
        --job-dir=${GCS_BUCKET}/jobs \
        --module-name=${SRC_PATH}.task \
        --package-path=${SRC_PATH}/ \
        --region=${REGION}

# Train on processed data.
elif [[ ${1} == "train" ]]; then

    if [[ ${EXECUTOR} == "local" ]]; then
        # Run ML Engine training locally
        gcloud ml-engine local train \
            --package-path ${SRC_PATH} \
            --module-name ${SRC_PATH}.task

    else
        # Run ML Engine training in the Google Cloud.
        gcloud ml-engine jobs submit training \
            ${JOB_NAME} \
            --module-name ${SRC_PATH}.task
    fi


# Predict on created model.
elif [[ ${1} == "predict" ]]; then

    # TODO: The directories below are still hardcoded. Values from config.yaml should be used.
    if [[ ${EXECUTOR} == "local" ]]; then
        # Run ML Engine prediction locally. Retrieves the latest model from the data directory.
        gcloud ml-engine local predict \
            --model-dir=`ls -d ./models/export/${SRC_PATH}/*/ | tail -n 1` \
            --json-instances=data/processed/test.json \
            | sed -E 's/(\[|\])//g' | sed -E 's/(  )+/,/g' > data/predictions/predictions-out-${DATE}.csv

    else
        # Run ML Engine prediction in the Google Cloud.
        gcloud ml-engine jobs submit prediction \
            ${JOB_NAME} \
            --model=elo \
            --data-format=text \
            --input-paths=${GCS_BUCKET}/data/processed/test.json \
            --output-path=${GCS_BUCKET}/data/prediction \
            --region=${REGION}
    fi

fi
