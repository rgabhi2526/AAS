#!/usr/bin/env bash
# Run AAS on all 13 datasets, 4 runs each (52 total jobs).
# Run this on Colab/GPU machine after cloning SpCL.
#
# Usage:
#   bash experiments/run_all.sh data/raw experiments/results

set -e

DATA_ROOT=${1:-data/raw}
OUTPUT_DIR=${2:-experiments/results}
CONFIG=experiments/configs/aas.yaml

DATASETS=(
  BelugaID
  CowDataset
  FriesianCattle2015
  GiraffeZebraID
  HyenaID2022
  IPanda50
  LeopardID2022
  MacaqueFaces
  NyalaData
  SealID
  WhaleSharkID
  HappyWhale
  ELPephants
)

mkdir -p "$OUTPUT_DIR"

for dataset in "${DATASETS[@]}"; do
  for run in 0 1 2 3; do
    out_file="${OUTPUT_DIR}/${dataset}_run${run}.json"
    if [ -f "$out_file" ]; then
      echo "SKIP: $out_file already exists"
      continue
    fi
    echo ""
    echo ">>> Dataset=$dataset | Run=$run"
    python3 experiments/train_aas.py \
      --config "$CONFIG" \
      --dataset "$dataset" \
      --data-root "$DATA_ROOT" \
      --output-dir "$OUTPUT_DIR" \
      --run-id "$run"
  done
done

echo ""
echo "All runs complete. Aggregating results..."
python3 experiments/aggregate_results.py "$OUTPUT_DIR"
