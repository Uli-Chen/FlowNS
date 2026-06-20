#!/bin/bash
# FlowNS staged experiment runner (v3, 2026-06-16; GRPO removed).
# Wraps `python -m src.pilot_runner run` with stage presets, logging, and a
# results summary. Stage definitions follow CLAUDE.md "Experiment focus".
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
RESULTS_DIR="$PROJECT_DIR/results/pilot"
LOG_DIR="$RESULTS_DIR/logs"

DATASET="mind"
SEED=""
TAG=""
DRY_RUN=0
SET_ARGS=()
TARGETS=()

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_experiments.sh [options] STAGE_OR_EXPERIMENT...

Stages (expand to experiment lists; see CLAUDE.md "Experiment focus"):
  backbone    S0_m0_lgcn S0_dns_lgcn         (F1: backbone health + headroom)
  s0          S0_m0_lgcn S0_dns_lgcn         (baseline + paired M0 + DNS control)
  s1          S1_flow_lgcn                   (F2: flow realness/bridge diagnostics)
  s3          S3_{rand,exposed,flow,cont}_lgcn (F2: ranking vs paired controls)
  lgcn-chain  S0..S3 LightGCN path in dependency order
  smoke       tiny end-to-end harness check (2-epoch, no pointer/cache pollution)

Any other token is passed through as an experiment name from
configs/experiments.yaml (see: python -m src.pilot_runner list).

Options:
  --dataset NAME    Dataset (default: mind; main conclusions are MIND-only)
  --seed N          Seed override; result files get an _sN suffix
  --tag TAG         Result-file suffix, e.g. for sweeps (--set + --tag pairs)
  --set KEY=VALUE   Config override, repeatable, forwarded to pilot_runner
  --dry-run         Print the commands without running
  -h, --help        This help

Examples:
  bash scripts/run_experiments.sh smoke
  bash scripts/run_experiments.sh s0 s1
  bash scripts/run_experiments.sh vae-chain
  bash scripts/run_experiments.sh S3_flowns_vae --seed 2021
  for g in 0.5 1 2 3; do
    bash scripts/run_experiments.sh S3_flowns_vae --set reward_gamma=$g --tag g$g
  done
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --seed)    SEED="$2"; shift 2 ;;
        --tag)     TAG="$2"; shift 2 ;;
        --set)     SET_ARGS+=("$2"); shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*)        echo "Unknown option: $1"; usage; exit 1 ;;
        *)         TARGETS+=("$1"); shift ;;
    esac
done

if [ ${#TARGETS[@]} -eq 0 ]; then
    usage
    exit 1
fi

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    exit 1
fi

cd "$PROJECT_DIR"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Smoke overrides: self-contained (own rec model, own flow cache, no M0
# pointer writes), tiny epochs. Validates every phase on real data in minutes.
SMOKE_SETS=(
    "epochs=2" "stopping_step=2" "flow_pretrain_epochs=2"
    "joint_rec_epochs=2" "diagnostic_sample_users=200" "quality_sample_users=200"
    "use_paired_m0=false" "save_rec_checkpoint_pointer=false"
    "flow_checkpoint_path=results/pilot/smoke_flow.pt"
    "flow_ref_checkpoint_path=results/pilot/smoke_flow_ref.pt"
)

expand_stage() {
    case "$1" in
        backbone)   echo "S0_m0_lgcn S0_dns_lgcn" ;;
        s0)         echo "S0_m0_lgcn S0_dns_lgcn" ;;
        s1)         echo "S1_flow_lgcn" ;;
        s3)         echo "S3_rand_lgcn S3_exposed_lgcn S3_flow_lgcn S3_cont_lgcn" ;;
        lgcn-chain) echo "S0_m0_lgcn S1_flow_lgcn S3_rand_lgcn S3_exposed_lgcn S3_flow_lgcn S3_cont_lgcn" ;;
        smoke)      echo "__SMOKE__" ;;
        *)          echo "$1" ;;
    esac
}

run_one() {
    local experiment=$1; shift
    local extra_sets=("$@")
    local suffix=""
    [ -n "$SEED" ] && suffix="${suffix}_s${SEED}"
    [ -n "$TAG" ] && suffix="${suffix}_${TAG}"
    local logfile="$LOG_DIR/${experiment}_${DATASET//[^A-Za-z0-9_]/_}${suffix}.log"

    local cmd=("$PYTHON" -m src.pilot_runner run --experiment "$experiment" --dataset "$DATASET")
    [ -n "$SEED" ] && cmd+=(--seed "$SEED")
    [ -n "$TAG" ] && cmd+=(--tag "$TAG")
    local s
    for s in "${SET_ARGS[@]:-}"; do [ -n "$s" ] && cmd+=(--set "$s"); done
    for s in "${extra_sets[@]:-}"; do [ -n "$s" ] && cmd+=(--set "$s"); done

    echo ""
    echo "============================================"
    echo "  $experiment  (dataset=$DATASET${SEED:+ seed=$SEED}${TAG:+ tag=$TAG})"
    echo "  log: $logfile"
    echo "============================================"
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  DRY-RUN:'; printf ' %q' "${cmd[@]}"; printf '\n'
        return 0
    fi
    if "${cmd[@]}" >"$logfile" 2>&1; then
        echo "  PASSED"
        tail -n 3 "$logfile" | sed 's/^/  | /'
    else
        echo "  FAILED (tail of $logfile):"
        tail -n 15 "$logfile" | sed 's/^/  | /'
        FAILED+=("$experiment")
    fi
}

FAILED=()
START_TIME=$(date +%s)

for target in "${TARGETS[@]}"; do
    for experiment in $(expand_stage "$target"); do
        if [ "$experiment" = "__SMOKE__" ]; then
            TAG_SAVED="$TAG"; TAG="smoke"
            for exp in S0_m0_lgcn S1_flow_lgcn S3_flow_lgcn S3_cont_lgcn; do
                run_one "$exp" "${SMOKE_SETS[@]}"
            done
            TAG="$TAG_SAVED"
        else
            run_one "$experiment"
        fi
    done
done

END_TIME=$(date +%s)
echo ""
echo "Done in $((END_TIME - START_TIME))s. Results: $RESULTS_DIR"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED experiments: ${FAILED[*]}"
fi

if [ "$DRY_RUN" -eq 0 ]; then
    echo ""
    "$PYTHON" scripts/summarize_results.py --dataset "$DATASET" || true
fi

[ ${#FAILED[@]} -eq 0 ]
