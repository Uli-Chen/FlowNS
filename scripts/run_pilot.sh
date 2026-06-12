#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
RESULTS_DIR="$PROJECT_DIR/results/pilot"
LOG_DIR="$RESULTS_DIR/logs"

DATASET="mind"
SEED=""
STAGES=()
SET_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_pilot.sh [options] [STAGE...]

Stages (M0/M1/M2 trunk + core controls):
  LightGCN line:
    M0     LightGCN baseline (paired-checkpoint source for M1/M2/M2b)
    M1     LightGCN pretrain + flow CFM pretrain + generation quality check
    M2     Full FlowNS (flow pretrain + GRPO + joint)
    M2b    Flow negatives without GRPO (isolates the GRPO contribution)
  MultiVAE line:
    M0vae  MultiVAE baseline (paired-checkpoint source for M2vae/M2vaeR)
    M2vae  MultiVAE + exposed-flow negatives, no GRPO
    M2vaeR MultiVAE + random-negative control (no flow, no GRPO)
  all    Run the M0/M1/M2 trunk above (sequential order is dependency-safe; see note)

  Round 10 experimental arm (bridge fix; not in `all`):
    M2vaeB MultiVAE + exposed-flow + boundary_topk FN-safe hardness mapping
           (tests vs M2vae nearest +0% and M2vaeR random +2.2%; reuses cached flow)

Note: the paired-M0 checkpoint pointer is keyed by dataset, not by model, so
the LightGCN and MultiVAE lines share it. In a single `all` run the LightGCN
trunk (M0..M2b) finishes before M0vae overwrites the pointer, so it is safe.
If you re-run a single LightGCN stage after the MultiVAE line, re-run M0 first.

Removed variants (M2c-M2af exploration, M3/M4/M5 ablations) and the reasoning
are documented in refine-logs/round-8-cleanup-and-reflection.md.

Options:
  --dataset DATASET       mind, ml-100k, yelp-2018, amazon-books, gowalla-merged
  --stage STAGE[,STAGE]   Stage list; can be repeated
  --seed SEED             Override seed
  --set KEY=VALUE         Override experiment config; can be repeated
  -h, --help              Show this help

Examples:
  bash scripts/run_pilot.sh --dataset mind --stage M0
  bash scripts/run_pilot.sh --dataset mind --stage M1,M2,M2b
  bash scripts/run_pilot.sh --dataset mind --stage M0vae,M2vae,M2vaeR
EOF
}

add_stages() {
    local value=$1
    local old_ifs=$IFS
    IFS=','
    read -ra parts <<< "$value"
    IFS=$old_ifs
    for part in "${parts[@]}"; do
        if [ -n "$part" ]; then
            STAGES+=("$part")
        fi
    done
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --stage|--stages)
            add_stages "$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --set)
            SET_ARGS+=("$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            add_stages "$1"
            shift
            ;;
    esac
done

if [ ${#STAGES[@]} -eq 0 ]; then
    STAGES=(all)
fi

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Create venv first, then install torch and recbole."
    exit 1
fi

cd "$PROJECT_DIR"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

$PYTHON -c "import torch, recbole; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')" >/dev/null

SAFE_DATASET="${DATASET//[^A-Za-z0-9_]/_}"

common_args() {
    printf '%s\n' --dataset "$DATASET"
    if [ -n "$SEED" ]; then
        printf '%s\n' --seed "$SEED"
    fi
    for item in "${SET_ARGS[@]}"; do
        printf '%s\n' --set "$item"
    done
}

run_experiment() {
    local experiment=$1
    local logfile="$LOG_DIR/${experiment}_${SAFE_DATASET}.log"
    local cmd=("$PYTHON" -m src.pilot_runner run --experiment "$experiment")

    while IFS= read -r arg; do cmd+=("$arg"); done < <(common_args)

    echo ""
    echo "============================================"
    echo "  $experiment / dataset=$DATASET"
    echo "  Log: $logfile"
    echo "============================================"

    if "${cmd[@]}" 2>&1 | tee "$logfile"; then
        echo "  $experiment: PASSED"
    else
        echo "  $experiment: FAILED (see $logfile)"
    fi
}

expand_stage() {
    case "$1" in
        all|ALL)
            # Dependency-safe order: the LightGCN trunk (M0..M2b) consumes the
            # per-dataset M0 pointer before M0vae overwrites it for the
            # MultiVAE line.
            printf '%s\n' M0 M1 M2 M2b M0vae M2vae M2vaeR
            ;;
        *)
            printf '%s\n' "$1"
            ;;
    esac
}

run_stage() {
    case "$1" in
        M0|m0)   run_experiment M0_baseline ;;
        M0vae|m0vae|M0VAE) run_experiment M0_multivae_baseline ;;
        M1|m1)   run_experiment M1_flow_pretrain ;;
        M2|m2)   run_experiment M2_flowns_full ;;
        M2b|m2b|M2B) run_experiment M2b_flow_no_grpo ;;
        M2vae|m2vae|M2VAE) run_experiment M2_multivae_flow_no_grpo ;;
        M2vaeR|m2vaer|M2VAER) run_experiment M2_multivae_random_neg_control ;;
        M2vaeB|m2vaeb|M2VAEB) run_experiment M2vae_boundary ;;
        *)
            echo "Unknown stage: $1"
            usage
            exit 1
            ;;
    esac
}

START_TIME=$(date +%s)
echo "FlowNS experiments: dataset=$DATASET stages=${STAGES[*]}"

for stage in "${STAGES[@]}"; do
    while IFS= read -r expanded; do
        run_stage "$expanded"
    done < <(expand_stage "$stage")
done

END_TIME=$(date +%s)
echo ""
echo "Completed in $((END_TIME - START_TIME))s"
echo "Results: $RESULTS_DIR"
echo "Logs:    $LOG_DIR"

echo ""
echo "--- Results Summary ---"
for f in "$RESULTS_DIR"/M*.json; do
    if [ -f "$f" ]; then
        "$PYTHON" - "$f" <<'PY' || true
import json
import sys

path = sys.argv[1]
with open(path) as fh:
    data = json.load(fh)
print(f"\n{path.split('/')[-1]}:")
test = data.get('test_result')
if isinstance(test, dict):
    print(f"  Recall@20={test.get('recall@20', test.get('recall@10', '?'))}, "
          f"NDCG@20={test.get('ndcg@20', test.get('ndcg@10', '?'))}")
if 'fn_rate' in data:
    print(f"  FN rate={data['fn_rate']}")
PY
    fi
done
