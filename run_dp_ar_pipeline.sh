#!/bin/bash
# ========================================
# Decision Point AR (Proposal C) Pipeline
# ========================================
# 
# Run on wsa workstation with:
#   bash run_dp_ar_pipeline.sh
#
# Prerequisites:
#   - paths_graph.npz (training + test)
#   - road_graph.npz

set -e

# Paths (adjust as needed)
DATA_DIR="/data/SFM/v3_data"
TRAIN_PATHS="${DATA_DIR}/detroit_train_paths_graph.npz"
TEST_PATHS="${DATA_DIR}/detroit_test_paths_graph.npz"
ROAD_GRAPH="${DATA_DIR}/detroit_road_graph.npz"
OUT_BASE="/data/SFM/v3_exps/dp_ar"

# Config
MIN_CHOICE_COUNT=2
MIN_OUT_DEGREE=2
BATCH_SIZE=256
N_EPOCHS=50
LR=0.001
MAX_CANDIDATES=32
SEED=42

echo "======================================"
echo "Step 1: Build Decision Point Graph"
echo "======================================"

python -m src.data.road_graph.build_decision_point_graph \
    --paths_graph_npz "${TRAIN_PATHS}" \
    --road_graph_npz "${ROAD_GRAPH}" \
    --out_dir "${OUT_BASE}/dp_graph" \
    --min_choice_count ${MIN_CHOICE_COUNT} \
    --min_out_degree ${MIN_OUT_DEGREE} \
    --seed ${SEED}

DP_GRAPH="${OUT_BASE}/dp_graph/decision_point_graph.npz"

echo ""
echo "======================================"
echo "Step 2: Train Decision Point AR Model"
echo "======================================"

python -m src.training.train_graph_ar_decision_point \
    --dp_graph_npz "${DP_GRAPH}" \
    --out_dir "${OUT_BASE}/train" \
    --batch_size ${BATCH_SIZE} \
    --n_epochs ${N_EPOCHS} \
    --lr ${LR} \
    --max_candidates ${MAX_CANDIDATES} \
    --seed ${SEED} \
    --device cuda

MODEL_PATH="${OUT_BASE}/train/model.pt"

echo ""
echo "======================================"
echo "Step 3: Evaluate on Test Set (Greedy)"
echo "======================================"

python -m src.training.sample_graph_ar_decision_point \
    --model_path "${MODEL_PATH}" \
    --dp_graph_npz "${DP_GRAPH}" \
    --road_graph_npz "${ROAD_GRAPH}" \
    --paths_graph_npz "${TEST_PATHS}" \
    --out_dir "${OUT_BASE}/eval_greedy" \
    --max_dp_steps 30 \
    --top_k 1 \
    --seed ${SEED} \
    --device cuda

echo ""
echo "======================================"
echo "Step 4: Evaluate with Top-3 Sampling"
echo "======================================"

python -m src.training.sample_graph_ar_decision_point \
    --model_path "${MODEL_PATH}" \
    --dp_graph_npz "${DP_GRAPH}" \
    --road_graph_npz "${ROAD_GRAPH}" \
    --paths_graph_npz "${TEST_PATHS}" \
    --out_dir "${OUT_BASE}/eval_top3" \
    --max_dp_steps 30 \
    --top_k 3 \
    --temperature 0.8 \
    --seed ${SEED} \
    --device cuda

echo ""
echo "======================================"
echo "Pipeline Complete!"
echo "======================================"
echo "Results in: ${OUT_BASE}"
