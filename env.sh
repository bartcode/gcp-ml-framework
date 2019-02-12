#!/usr/bin/env bash

DATE=`date '+%Y%m%d_%H%M%S'`
SCALE_TIER=STANDARD_1
REGION=europe-west4

HPTUNING_CONFIG=./hyperparam.yaml
GCS_JOB_DIR=gs://some-bucket/jobs
