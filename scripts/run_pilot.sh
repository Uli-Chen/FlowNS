#!/bin/bash
# =============================================================================
# FlowNS Pilot Experiments on MINDsmall
# =============================================================================
#
# Experiments:
#   M0: LightGCN baseline
#   M1: Flow pretrain + generation quality check
#   M2: FlowNS full 3-phase pipeline
#   M2b: Flow pretrain + no GRPO, paired with M0 checkpoint
#   M3: Reward ablation (Unshaped R=W, NoRL)
#   M3b: RNS pretrain + RNS finetune control (no flow training)
#   M4: KL ablation (β ∈ {0, 0.1, 0.5})
#   M5: FN guarantee verification (theoretical bound vs actual)
#
# Usage:
#   cd /home/chen/workspace/flowns
#   bash scripts/run_pilot.sh          # run all
#   bash scripts/run_pilot.sh M0       # run specific milestone
#   bash scripts/run_pilot.sh M0 M2    # run multiple milestones
#   bash scripts/run_pilot.sh M3b      # run RNS control only
#
# Results saved to: results/pilot/
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
RESULTS_DIR="$PROJECT_DIR/results/pilot"
LOG_DIR="$PROJECT_DIR/results/pilot/logs"

cd "$PROJECT_DIR"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Check Python and dependencies
echo "============================================"
echo "  FlowNS Pilot Experiment Runner"
echo "============================================"
echo "Project dir: $PROJECT_DIR"
echo "Python:      $PYTHON"
echo ""

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Please create venv: python3 -m venv .venv && .venv/bin/pip install torch recbole ..."
    exit 1
fi

$PYTHON -c "import torch, recbole; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')" 2>/dev/null || {
    echo "ERROR: Missing dependencies. Install torch and recbole in the venv."
    exit 1
}

# Determine which milestones to run
if [ $# -eq 0 ]; then
    MILESTONES="M0 M1 M2 M2b M3 M3b M4 M5"
else
    MILESTONES="$@"
fi

run_milestone() {
    local name=$1
    local script=$2
    local logfile="$LOG_DIR/${name}.log"

    echo ""
    echo "============================================"
    echo "  $name: Starting..."
    echo "  Log: $logfile"
    echo "============================================"

    if $PYTHON "$script" 2>&1 | tee "$logfile"; then
        echo "  ✓ $name: PASSED"
    else
        echo "  ✗ $name: FAILED (see $logfile)"
        echo "  Continuing to next milestone..."
    fi
}

TOTAL_START=$(date +%s)

for M in $MILESTONES; do
    case $M in
        M0)
            echo ""
            echo "============================================"
            echo "  M0: LightGCN Baseline on MIND"
            echo "  Goal: Establish baseline Recall@20/NDCG@20"
            echo "============================================"
            run_milestone "M0_baseline" "$SCRIPT_DIR/pilot_m0_baseline.py"
            ;;
        M1)
            echo ""
            echo "============================================"
            echo "  M1: Flow Pretrain + Generation Quality"
            echo "  Goal: FN rate < 5%, meaningful distribution"
            echo "============================================"
            run_milestone "M1_flow_pretrain" "$SCRIPT_DIR/pilot_m1_flow.py"
            ;;
        M2)
            echo ""
            echo "============================================"
            echo "  M2: FlowNS Full Pipeline"
            echo "  Uses paired M0 checkpoint"
            echo "  Goal: FlowNS > LightGCN(Uniform)"
            echo "============================================"
            run_milestone "M2_flowns_full" "$SCRIPT_DIR/pilot_m2_flowns.py"
            ;;
        M2b)
            echo ""
            echo "============================================"
            echo "  M2b: Flow Pretrain + No GRPO"
            echo "  Uses paired M0 checkpoint"
            echo "  Goal: Isolate flow negatives without GRPO"
            echo "============================================"
            run_milestone "M2b_flow_no_grpo" "$SCRIPT_DIR/pilot_m2b_flow_no_grpo.py"
            ;;
        M3)
            echo ""
            echo "============================================"
            echo "  M3: Reward Ablation"
            echo "  Variants: Unshaped (R=W), NoRL"
            echo "  Goal: Full > Unshaped > NoRL"
            echo "============================================"
            run_milestone "M3_reward_ablation" "$SCRIPT_DIR/pilot_m3_reward_ablation.py"
            ;;
        M3b)
            echo ""
            echo "============================================"
            echo "  M3b: RNS Pretrain + RNS Finetune"
            echo "  No flow pretrain or GRPO flow updates"
            echo "  Uses uniform random negatives for finetune"
            echo "  Goal: FlowNS-Full > RNS-Finetune"
            echo "============================================"
            run_milestone "M3b_random_neg" "$SCRIPT_DIR/pilot_m3b_random_neg.py"
            ;;
        M4)
            echo ""
            echo "============================================"
            echo "  M4: KL (Beta) Ablation"
            echo "  β ∈ {0, 0.1, 0.5}"
            echo "  Goal: β=0 causes FN spike"
            echo "============================================"
            run_milestone "M4_kl_ablation" "$SCRIPT_DIR/pilot_m4_kl_ablation.py"
            ;;
        M5)
            echo ""
            echo "============================================"
            echo "  M5: FN Guarantee Verification"
            echo "  Uses MIND exposure data"
            echo "  Goal: Actual FN < theoretical bound"
            echo "============================================"
            run_milestone "M5_fn_guarantee" "$SCRIPT_DIR/pilot_m5_fn_guarantee.py"
            ;;
        *)
            echo "Unknown milestone: $M (valid: M0 M1 M2 M2b M3 M3b M4 M5)"
            ;;
    esac
done

TOTAL_END=$(date +%s)
ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "============================================"
echo "  All pilot experiments completed"
echo "  Total time: ${ELAPSED}s ($((ELAPSED/60))m)"
echo "  Results:    $RESULTS_DIR/"
echo "  Logs:       $LOG_DIR/"
echo "============================================"

# Print summary if result files exist
echo ""
echo "--- Results Summary ---"
for f in "$RESULTS_DIR"/M*.json; do
    if [ -f "$f" ]; then
        echo ""
        echo "$(basename "$f"):"
        $PYTHON -c "
import json, sys
with open('$f') as fh:
    d = json.load(fh)
    if 'test_result' in d:
        tr = d['test_result']
        r20 = tr.get('recall@20', tr.get('recall@10', '?'))
        n20 = tr.get('ndcg@20', tr.get('ndcg@10', '?'))
        print(f'  Recall@20={r20}, NDCG@20={n20}')
    if 'fn_rate' in d:
        print(f'  FN rate={d[\"fn_rate\"]}')
    if 'fn_ref' in d:
        print(f'  FN ref={d[\"fn_ref\"]}')
    if 'results' in d and isinstance(d['results'], list):
        for r in d['results']:
            if 'beta' in r:
                print(f'  β={r[\"beta\"]}: FN_actual={r[\"fn_actual\"]:.6f}, bound={r[\"fn_bound\"]:.6f}, tightness={r[\"tightness_ratio\"]:.4f}')
" 2>/dev/null || true
    fi
done
