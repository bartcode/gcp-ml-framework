#!/usr/bin/env bash

SRC_PATH=src

DATE=`date '+%Y%m%d_%H%M%S'`
SCALE_TIER=STANDARD_1
REGION=europe-west4

HPTUNING_CONFIG=./hyperparam.yaml
GCS_BUCKET=gs://some-bucket/jobs
