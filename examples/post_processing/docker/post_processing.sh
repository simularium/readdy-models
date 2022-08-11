#!/bin/bash

# Download trajectory files
case ${SIMULATION_TYPE} in
	AWS)
		FILE_INDEX=${AWS_BATCH_JOB_ARRAY_INDEX:-0}
	;;
	LOCAL)
        FILE_INDEX=$JOB_ARRAY_INDEX
	;;
esac
S3_FILE_PATH="${S3_INPUT_URL}outputs/"
echo $S3_FILE_PATH
FILE_NAME="file"
aws s3 cp . $S3_FILE_PATH --exclude "*" --include "actin_twist_bend_dihedral_strength10_tangent_free_9.h5" --recursive --dryrun

# Post-process files
python post_processing.py $FILE_NAME

# Upload output files
case ${SIMULATION_TYPE} in
	AWS)
		aws s3 sync ./outputs $OUTPUT_FILE_PATH
	;;
	LOCAL)
		cd outputs
		cp *.h5 $OUTPUT_FILE_PATH
		cp *.simularium $OUTPUT_FILE_PATH
		cp *.dat $OUTPUT_FILE_PATH
	;;
esac
