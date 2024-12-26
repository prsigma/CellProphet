#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

datasets=("mHSC-E" "mHSC-GM" "mHSC-L" )
output_dir="./Result_4_perturbation/output"
# prior_ratio=("0.0" "0.05" "0.1" "0.15" "0.2" "0.25" "0.3" "0.35" "0.4")

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"

    species="mouse"
    tf_path="/home/pengrui/STGRN/BEELINE-Networks/mouse-tfs.csv"
   
    # 如果dataset以mHSC开头，则跳过
    gt_name="mHSC"
    /home/pengrui/mambaforge/envs/grn/bin/python main.py --data_source BEELINE --dataset_name ${dataset} --data_path "BEELINE-data/inputs/scRNA-Seq/${dataset}/ExpressionData.csv" \
    --output_dir ${output_dir} --gt_path "BEELINE-Networks/Networks/${species}/${gt_name}-ChIP-seq-network.csv" --time_info "BEELINE-data/inputs/scRNA-Seq/${dataset}/PseudoTime.csv" \
    --gene_order BEELINE-data/inputs/scRNA-Seq/${dataset}/GeneOrdering.csv --tf_path ${tf_path} --species ${species} --logFC_path "BEELINE-data/inputs/scRNA-Seq/${dataset}/DEgenes_MAST_sp4_PseudoTime.csv" --cuda_index 0
done