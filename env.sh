#!/usr/bin/env bash

export SRC_PATH=src
export MODEL_NAME=model  # Should be same as config.yml (model.name)

export DATE=`date '+%Y%m%d_%H%M%S'`
export SCALE_TIER=BASIC
export REGION=europe-west4

export HPTUNING_CONFIG=./hyperparam.yml
export GCS_BUCKET=gs://some-bucket/
