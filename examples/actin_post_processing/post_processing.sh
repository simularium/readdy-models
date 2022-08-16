#!/bin/bash

# set up
OUTPUT_FILE_PATH="${S3_INPUT_URL}outputs/"
PARAMS_COL_INDEX=${AWS_BATCH_JOB_ARRAY_INDEX:-0}

# Run 
python post_processing.py $PARAMS_COL_INDEX

# Save output files
aws s3 sync ./outputs $OUTPUT_FILE_PATH
