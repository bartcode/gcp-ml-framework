#!/usr/bin/env bash

SRC_PATH=src
MODEL_NAME=model  # Should be same as config.yml (model.name)

DATE=`date '+%Y%m%d_%H%M%S'`
SCALE_TIER=BASIC
REGION=europe-west4

HPTUNING_CONFIG=./hyperparam.yaml
GCS_BUCKET=gs://some-bucket/jobs
