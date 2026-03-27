#!/bin/bash
DATASET=${1:-all}
SEQ_LEN=${2:-10}
OUTPUT_FRAME=${3:-1}
OUTPUT_DIR="seq${SEQ_LEN}_frame${OUTPUT_FRAME}"

DATASETS=("bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex")

if [ "$DATASET" == "all" ]; then
    for ds in "${DATASETS[@]}"; do
        OUTPUT_PATH="/root/autodl-tmp/outputs/futuregs/dnerf/${ds}/${OUTPUT_DIR}"
        python train.py -s "/root/autodl-tmp/datas/dnerf_predict(9:1)/$ds" -m "$OUTPUT_PATH" --eval --is_blender --seq_len $SEQ_LEN --output_frame $OUTPUT_FRAME && \
        python render.py -s "/root/autodl-tmp/datas/dnerf_predict(9:1)/$ds" -m "$OUTPUT_PATH" --eval --is_blender --iteration -1 --seq_len $SEQ_LEN --output_frame $OUTPUT_FRAME && \
        python metrics.py --model_paths "$OUTPUT_PATH"
    done
else
    OUTPUT_PATH="/root/autodl-tmp/outputs/futuregs/dnerf/${DATASET}/${OUTPUT_DIR}"
    python train.py -s "/root/autodl-tmp/datas/dnerf_predict(9:1)/$DATASET" -m "$OUTPUT_PATH" --eval --is_blender --seq_len $SEQ_LEN --output_frame $OUTPUT_FRAME && \
    python render.py -s "/root/autodl-tmp/datas/dnerf_predict(9:1)/$DATASET" -m "$OUTPUT_PATH" --eval --is_blender --iteration -1 --seq_len $SEQ_LEN --output_frame $OUTPUT_FRAME && \
    python metrics.py --model_paths "$OUTPUT_PATH"
fi
