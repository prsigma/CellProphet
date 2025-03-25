#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

datasets=("mESC" "mHSC-GM" "mHSC-L" "mHSC-E")
output_dir="./output"

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"

    species='mouse'
    tf_path="./TF/mouse-tfs.csv"
   
    if [[ $dataset == mHSC* ]]; then
        gt_name="mHSC"
        python main.py --data_source BEELINE --dataset_name ${dataset} --data_path "BEELINE-data/${dataset}/ExpressionData.csv" \
    --output_dir ${output_dir} --gt_path "BEELINE-Networks/${gt_name}-ChIP-seq-network.csv" --time_info "BEELINE-data/${dataset}/PseudoTime.csv" \
    --gene_order BEELINE-data/${dataset}/GeneOrdering.csv --tf_path ${tf_path} --species ${species} --logFC_path "BEELINE-data/${dataset}/DEgenes_MAST_sp4_PseudoTime.csv" --cuda_index 0 --baseline
    else
    python main.py --data_source BEELINE --dataset_name ${dataset} --data_path "BEELINE-data/${dataset}/ExpressionData.csv" \
    --output_dir ${output_dir} --gt_path "BEELINE-Networks/${dataset}-ChIP-seq-network.csv" --time_info "BEELINE-data/${dataset}/PseudoTime.csv" \
    --gene_order BEELINE-data/${dataset}/GeneOrdering.csv --tf_path ${tf_path} --species ${species} --logFC_path "BEELINE-data/${dataset}/DEgenes_MAST_sp4_PseudoTime.csv" --cuda_index 0 --baseline
    fi
done