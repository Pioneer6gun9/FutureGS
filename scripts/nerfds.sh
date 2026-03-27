#!/bin/bash
DATASET=${1:-all}
SEQ_LEN=${2:-10}
OUTPUT_FRAME=${3:-1}
OUTPUT_DIR="seq${SEQ_LEN}_frame${OUTPUT_FRAME}"

DATASETS=("as_novel_view" "basin_novel_view" "bell_novel_view" "cup_novel_view" "plate_novel_view" "press_novel_view" "sieve_novel_view")

if [ "$DATASET" == "all" ]; then
    for ds in "${DATASETS[@]}"; do
        OUTPUT_PATH="/root/autodl-tmp/outputs/futuregs/nerfds/${ds}/${OUTPUT_DIR}"
        python train.py -s "/root/autodl-tmp/datas/NeRF-DS_predict(9:1)/$ds" -m "$OUTPUT_PATH" --eval --iterations 20000 --seq_len $SEQ_LEN --output_frame $OUTPUT_FRAME && \
        python render.py -s "/root/autodl-tmp/datas/NeRF-DS_predict(9:1)/$ds" -m "$OUTPUT_PATH" --eval --iteration -1 --seq_len $SEQ_LEN --output_frame $OUTPUT_FRAME && \
        python metrics.py --model_paths "$OUTPUT_PATH"
    done
else
    OUTPUT_PATH="/root/autodl-tmp/outputs/futuregs/nerfds/${DATASET}/${OUTPUT_DIR}"
    python train.py -s "/root/autodl-tmp/datas/NeRF-DS_predict(9:1)/$DATASET" -m "$OUTPUT_PATH" --eval --iterations 20000 --seq_len $SEQ_LEN --output_frame $OUTPUT_FRAME && \
    python render.py -s "/root/autodl-tmp/datas/NeRF-DS_predict(9:1)/$DATASET" -m "$OUTPUT_PATH" --eval --iteration -1 --seq_len $SEQ_LEN --output_frame $OUTPUT_FRAME && \
    python metrics.py --model_paths "$OUTPUT_PATH"
fi
