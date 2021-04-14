#!/bin/bash

echo "************ start"
# copy parameter file
case ${SIMULATION_TYPE} in
	AWS)
		INPUT_FILE_PATH="${S3_INPUT_URL}parameters/"
		OUTPUT_FILE_PATH="${S3_INPUT_URL}outputs/"
        PARAMS_COL_INDEX=$((${AWS_BATCH_JOB_ARRAY_INDEX:-0} + 1))
		INPUT_FILE_NAME="${PARAM_SET_NAME}.xlsx"
		aws s3 cp $INPUT_FILE_PATH$INPUT_FILE_NAME input.xlsx
	;;
	LOCAL)
		INPUT_FILE_PATH="/working/"
		OUTPUT_FILE_PATH="/working/"
        PARAMS_COL_INDEX=$((${JOB_ARRAY_INDEX} + 1))
		INPUT_FILE_NAME="${PARAM_SET_NAME}.xlsx"
		cp $INPUT_FILE_PATH$INPUT_FILE_NAME input.xlsx
	;;
esac

echo "************ pre run"
# Run the model
python actin.py input.xlsx $PARAMS_COL_INDEX ${PARAM_SET_NAME}
EXIT_CODE=$?
echo "************ post run"

# Save output files
if [ $EXIT_CODE -eq 0 ]
then
	case ${SIMULATION_TYPE} in
		AWS)
			aws s3 cp . $OUTPUT_FILE_PATH --recursive --exclude "*" --include "*.h5" --exclude "*/*"
			aws s3 cp . $OUTPUT_FILE_PATH --recursive --exclude "*" --include "*.simularium" --exclude "*/*"
            aws s3 sync ./checkpoints "${OUTPUT_FILE_PATH}checkpoints/"
        ;;
		LOCAL)
			cp *.h5 $OUTPUT_FILE_PATH
			cp *.simularium $OUTPUT_FILE_PATH
			cp checkpoints "${OUTPUT_FILE_PATH}" -r
		;;
	esac
fi

echo "************ end"