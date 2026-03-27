#!/bin/bash
DATASET=${1:-all}
SEQ_LEN=${2:-10}
OUTPUT_FRAME=${3:-1}
OUTPUT_DIR="seq${SEQ_LEN}_frame${OUTPUT_FRAME}"

DATASETS=("aleks-teapot" "americano" "broom2" "chickchicken" "cross-hands1" "cut-lemon1" "espresso" "hand1-dense-v2" "keyboard" "oven-mitts" "slice-banana" "split-cookie" "tamping" "torchocolate" "vrig-3dprinter" "vrig-chicken" "vrig-peel-banana")

if [ "$DATASET" == "all" ]; then
    for ds in "${DATASETS[@]}"; do
        OUTPUT_PATH="/root/autodl-tmp/outputs/futuregs/hypernerf/${ds}/${OUTPUT_DIR}"
        python train.py -s "/root/autodl-tmp/datas/Hyper-NeRF_predict(9:1)/$ds" -m "$OUTPUT_PATH" --eval --iterations 20000 --seq_len $SEQ_LEN --output_frame $OUTPUT_FRAME && \
        python render.py -s "/root/autodl-tmp/datas/Hyper-NeRF_predict(9:1)/$ds" -m "$OUTPUT_PATH" --eval --iteration -1 --seq_len $SEQ_LEN --output_frame $OUTPUT_FRAME && \
        python metrics.py --model_paths "$OUTPUT_PATH"
    done
else
    OUTPUT_PATH="/root/autodl-tmp/outputs/futuregs/hypernerf/${DATASET}/${OUTPUT_DIR}"
    python train.py -s "/root/autodl-tmp/datas/Hyper-NeRF_predict(9:1)/$DATASET" -m "$OUTPUT_PATH" --eval --iterations 20000 --seq_len $SEQ_LEN --output_frame $OUTPUT_FRAME && \
    python render.py -s "/root/autodl-tmp/datas/Hyper-NeRF_predict(9:1)/$DATASET" -m "$OUTPUT_PATH" --eval --iteration -1 --seq_len $SEQ_LEN --output_frame $OUTPUT_FRAME && \
    python metrics.py --model_paths "$OUTPUT_PATH"
fi
