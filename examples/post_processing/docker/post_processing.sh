#!/bin/bash

# Download trajectory files
case ${SIMULATION_TYPE} in
	AWS)
		CONDITION_INDEX=${AWS_BATCH_JOB_ARRAY_INDEX:-0}
		OUTPUT_FILE_PATH="s3://readdy-working-bucket/outputs/"
	;;
	LOCAL)
        CONDITION_INDEX=$JOB_ARRAY_INDEX
		OUTPUT_FILE_PATH="/working/"
	;;
esac

# Post-process files
python post_processing.py $CONDITION_INDEX

# Upload output files
case ${SIMULATION_TYPE} in
	AWS)
		aws s3 sync ./outputs $OUTPUT_FILE_PATH
	;;
	LOCAL)
		cd outputs
		cp *.simularium $OUTPUT_FILE_PATH
		cp *.dat $OUTPUT_FILE_PATH
	;;
esac
