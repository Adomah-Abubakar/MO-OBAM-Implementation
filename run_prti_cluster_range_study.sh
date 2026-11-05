#!/usr/bin/env bash
# Run or resume an aggregate-only PRTI-GH cluster-range selection study.

set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 FROM_CLUSTER TO_CLUSTER [OPTIONS]" \
    "" \
    "Required options:" \
    "  --k N                       Hard minimum equivalence-class size" \
    "  --lambda-weight VALUE       Privacy weight from 0 through 1" \
    "  --population N              PSO population, at least 2" \
    "  --iterations N              PSO iterations, at least 1" \
    "  --seeds CSV                  Comparison seeds, e.g. 0,1,2,3,4,5,6,7,8,9" \
    "  --held-out-seed N            Unseen final-candidate seed" \
    "" \
    "Optional controls:" \
    "  --workers N                 Concurrent seed runs (default: 4)" \
    "  --worker-backend MODE       thread or process (default: thread)" \
    "  --max-new-runs N            Stop after N new checkpoints; rerun to resume" \
    "  --output-directory PATH     Override the configuration-labelled path" \
    "  --reuse-directory PATH      Import only exact matching aggregate checkpoints" \
    "  --no-reuse                  Import no historical checkpoints" \
    "  --retry-blocked             Retry prior no-feasible-particle checkpoints" \
    "  --dry-run                   Write the immutable plan without executing PSO" \
    "  --help                      Show this help" \
    "" \
    "Recommended full comparison for k=10:" \
    "  $0 1 148 --k 10 --lambda-weight 1 --population 100 --iterations 100 --seeds 0,1,2,3,4,5,6,7,8,9 --held-out-seed 11 --workers 4" \
    "" \
    "For k=20, the largest feasible endpoint is 74:" \
    "  $0 1 74 --k 20 --lambda-weight 1 --population 100 --iterations 100 --seeds 0,1,2,3,4,5,6,7,8,9 --held-out-seed 11 --workers 4" \
    "" \
    "This search is aggregate-only. It selects the lowest mean information " \
    "loss only among configurations passing every predeclared privacy and " \
    "feasibility check across every comparison seed."
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi
if (( $# < 2 )); then
  usage >&2
  exit 2
fi

from_text=$1
to_text=$2
shift 2
if [[ ! $from_text =~ ^[0-9]+$ || ! $to_text =~ ^[0-9]+$ ]]; then
  printf 'Error: FROM_CLUSTER and TO_CLUSTER must be positive integers.\n' >&2
  exit 2
fi
from_cluster=$((10#$from_text))
to_cluster=$((10#$to_text))
if (( from_cluster < 1 || to_cluster < from_cluster )); then
  printf 'Error: require 1 <= FROM_CLUSTER <= TO_CLUSTER.\n' >&2
  exit 2
fi

k=""
lambda_weight=""
population=""
iterations=""
seeds_text=""
held_out_seed=""
workers=4
worker_backend=thread
max_new_runs=""
output_directory=""
reuse_directory=""
no_reuse=false
retry_blocked=false
dry_run=false

while (( $# > 0 )); do
  case $1 in
    --k)
      if (( $# < 2 )) || [[ ! $2 =~ ^[0-9]+$ ]] || (( 10#$2 < 2 )); then
        printf 'Error: --k requires an integer of at least 2.\n' >&2
        exit 2
      fi
      k=$((10#$2)); shift 2
      ;;
    --lambda-weight)
      if (( $# < 2 )) || [[ ! $2 =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]]; then
        printf 'Error: --lambda-weight must be a number from 0 through 1.\n' >&2
        exit 2
      fi
      lambda_weight=$2; shift 2
      ;;
    --population)
      if (( $# < 2 )) || [[ ! $2 =~ ^[0-9]+$ ]] || (( 10#$2 < 2 )); then
        printf 'Error: --population requires an integer of at least 2.\n' >&2
        exit 2
      fi
      population=$((10#$2)); shift 2
      ;;
    --iterations)
      if (( $# < 2 )) || [[ ! $2 =~ ^[0-9]+$ ]] || (( 10#$2 < 1 )); then
        printf 'Error: --iterations requires a positive integer.\n' >&2
        exit 2
      fi
      iterations=$((10#$2)); shift 2
      ;;
    --seeds)
      if (( $# < 2 )) || [[ -z $2 ]]; then
        printf 'Error: --seeds requires a comma-separated list.\n' >&2
        exit 2
      fi
      seeds_text=$2; shift 2
      ;;
    --held-out-seed)
      if (( $# < 2 )) || [[ ! $2 =~ ^[0-9]+$ ]]; then
        printf 'Error: --held-out-seed requires a non-negative integer.\n' >&2
        exit 2
      fi
      held_out_seed=$((10#$2)); shift 2
      ;;
    --workers)
      if (( $# < 2 )) || [[ ! $2 =~ ^[0-9]+$ ]] || (( 10#$2 < 1 )); then
        printf 'Error: --workers requires a positive integer.\n' >&2
        exit 2
      fi
      workers=$((10#$2)); shift 2
      ;;
    --worker-backend)
      if (( $# < 2 )) || [[ $2 != thread && $2 != process ]]; then
        printf 'Error: --worker-backend must be thread or process.\n' >&2
        exit 2
      fi
      worker_backend=$2; shift 2
      ;;
    --max-new-runs)
      if (( $# < 2 )) || [[ ! $2 =~ ^[0-9]+$ ]] || (( 10#$2 < 1 )); then
        printf 'Error: --max-new-runs requires a positive integer.\n' >&2
        exit 2
      fi
      max_new_runs=$((10#$2)); shift 2
      ;;
    --output-directory)
      if (( $# < 2 )) || [[ -z $2 ]]; then
        printf 'Error: --output-directory requires a path.\n' >&2
        exit 2
      fi
      output_directory=$2; shift 2
      ;;
    --reuse-directory)
      if (( $# < 2 )) || [[ -z $2 ]]; then
        printf 'Error: --reuse-directory requires a path.\n' >&2
        exit 2
      fi
      if [[ $no_reuse == true ]]; then
        printf 'Error: --reuse-directory and --no-reuse are mutually exclusive.\n' >&2
        exit 2
      fi
      reuse_directory=$2; shift 2
      ;;
    --no-reuse)
      if [[ -n $reuse_directory ]]; then
        printf 'Error: --reuse-directory and --no-reuse are mutually exclusive.\n' >&2
        exit 2
      fi
      no_reuse=true; shift
      ;;
    --retry-blocked)
      retry_blocked=true; shift
      ;;
    --dry-run)
      dry_run=true; shift
      ;;
    --help|-h)
      usage; exit 0
      ;;
    *)
      printf 'Error: unknown option %q.\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

missing_parameters=()
[[ -n $k ]] || missing_parameters+=(--k)
[[ -n $lambda_weight ]] || missing_parameters+=(--lambda-weight)
[[ -n $population ]] || missing_parameters+=(--population)
[[ -n $iterations ]] || missing_parameters+=(--iterations)
[[ -n $seeds_text ]] || missing_parameters+=(--seeds)
[[ -n $held_out_seed ]] || missing_parameters+=(--held-out-seed)
if (( ${#missing_parameters[@]} > 0 )); then
  printf 'Error: required terminal parameters missing: %s\n' \
    "${missing_parameters[*]}" >&2
  exit 2
fi

maximum_clusters=$((1485 / k))
if (( to_cluster > maximum_clusters )); then
  printf 'Error: TO_CLUSTER=%d is impossible for 1,485 rows with k=%d; maximum is %d.\n' \
    "$to_cluster" "$k" "$maximum_clusters" >&2
  exit 2
fi

IFS=',' read -r -a seed_values <<< "$seeds_text"
declare -A seen_seeds=()
for seed_text in "${seed_values[@]}"; do
  if [[ ! $seed_text =~ ^[0-9]+$ ]]; then
    printf 'Error: every --seeds value must be a non-negative integer.\n' >&2
    exit 2
  fi
  seed=$((10#$seed_text))
  if [[ -n ${seen_seeds[$seed]:-} ]]; then
    printf 'Error: --seeds values must be unique; duplicate %d.\n' "$seed" >&2
    exit 2
  fi
  if (( seed == held_out_seed )); then
    printf 'Error: held-out seed %d must not appear in --seeds.\n' "$seed" >&2
    exit 2
  fi
  seen_seeds[$seed]=1
done

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_path=${PRTI_PYTHON:-"$script_directory/.venv/bin/python"}
if [[ ! -x $python_path ]]; then
  printf 'Error: Python interpreter not found at %s. Create .venv or set PRTI_PYTHON.\n' \
    "$python_path" >&2
  exit 1
fi

if [[ -z $output_directory ]]; then
  printf -v from_label '%03d' "$from_cluster"
  printf -v to_label '%03d' "$to_cluster"
  lambda_label=${lambda_weight//./p}
  seeds_label=${seeds_text//,/-}
  output_directory="$script_directory/evidence/prti-range-c${from_label}-to-c${to_label}-k${k}-lambda${lambda_label}-pop${population}-iter${iterations}-seeds${seeds_label}-heldout${held_out_seed}-v1"
fi

command=(
  "$python_path"
  "$script_directory/run_prti_experiments.py"
  dense-clusters
  --minimum-clusters "$from_cluster"
  --maximum-clusters "$to_cluster"
  --output-directory "$output_directory"
  --k "$k"
  --lambda-weight "$lambda_weight"
  --population "$population"
  --iterations "$iterations"
  --seeds "${seed_values[@]}"
  --held-out-seed "$held_out_seed"
  --workers "$workers"
  --worker-backend "$worker_backend"
)
if [[ $no_reuse == true ]]; then
  command+=(--no-reuse)
elif [[ -n $reuse_directory ]]; then
  command+=(--reuse-directory "$reuse_directory")
else
  command+=(--reuse-directory "$script_directory/evidence/prti-robustness-v1")
fi
[[ -z $max_new_runs ]] || command+=(--max-new-runs "$max_new_runs")
[[ $retry_blocked == false ]] || command+=(--retry-blocked)
[[ $dry_run == false ]] || command+=(--dry-run)

planned_configurations=$((to_cluster - from_cluster + 1))
planned_runs=$((planned_configurations * ${#seed_values[@]}))
printf 'PRTI-GH aggregate cluster-range study\n'
printf '  cluster counts: %d through %d\n' "$from_cluster" "$to_cluster"
printf '  k: %d\n' "$k"
printf '  lambda weight: %s\n' "$lambda_weight"
printf '  population: %d\n' "$population"
printf '  iterations: %d\n' "$iterations"
printf '  comparison seeds: %s\n' "$seeds_text"
printf '  held-out seed: %d (not run during selection)\n' "$held_out_seed"
printf '  planned aggregate runs: %d\n' "$planned_runs"
printf '  worker backend: %s\n' "$worker_backend"
printf '  output: %s\n' "$output_directory"
if [[ $no_reuse == true ]]; then
  printf '  checkpoint imports: disabled\n'
elif [[ -n $reuse_directory ]]; then
  printf '  checkpoint imports: exact matches from %s\n' "$reuse_directory"
else
  printf '  checkpoint imports: exact matches from %s\n' \
    "$script_directory/evidence/prti-robustness-v1"
fi
printf '  patient-level rows: disabled\n'
printf '  resume behavior: matching completed checkpoints are reused\n'

cd -- "$script_directory"
"${command[@]}"
