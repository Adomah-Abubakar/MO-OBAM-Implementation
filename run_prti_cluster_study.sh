#!/usr/bin/env bash
# Run or resume an aggregate-only PRTI-GH study for one cluster count.

set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 CLUSTER [OPTIONS]" \
    "" \
    "Options:" \
    "  --k N                       Required: hard minimum equivalence-class size" \
    "  --lambda-weight VALUE       Required: privacy weight from 0 through 1" \
    "  --population N              Required: PSO population, at least 2" \
    "  --iterations N              Required: PSO iterations, at least 1" \
    "  --seeds CSV                  Required: comparison seeds, e.g. 0,1,2,3" \
    "  --held-out-seed N            Required: unseen final-candidate seed" \
    "  --workers N                 Concurrent independent seed runs (default: 4)" \
    "  --worker-backend MODE       thread or process (default: thread)" \
    "  --output-directory PATH     Override the configuration-labelled output path" \
    "  --dry-run                   Write the plan without executing PSO" \
    "  --help                      Show this help" \
    "" \
    "Examples:" \
    "  $0 8 --k 10 --lambda-weight 1 --population 100 --iterations 100 --seeds 0,1,2,3,4,5,6,7,8,9 --held-out-seed 11" \
    "  $0 12 --k 15 --lambda-weight 0.1 --population 150 --iterations 200 --seeds 9 --held-out-seed 11" \
    "  $0 25 --k 10 --lambda-weight 1 --population 100 --iterations 100 --seeds 0,1,2 --held-out-seed 11 --workers 2 --dry-run" \
    "" \
    "The command writes aggregate metrics only, never patient-level rows."
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

cluster_text=$1
shift
if [[ ! $cluster_text =~ ^[0-9]+$ ]]; then
  printf 'Error: CLUSTER must be a positive integer; received %q.\n' "$cluster_text" >&2
  exit 2
fi
cluster=$((10#$cluster_text))
if (( cluster < 1 )); then
  printf 'Error: CLUSTER must be at least 1.\n' >&2
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
output_directory=""
dry_run=false
while (( $# > 0 )); do
  case $1 in
    --k)
      if (( $# < 2 )) || [[ ! $2 =~ ^[0-9]+$ ]] || (( 10#$2 < 2 )); then
        printf 'Error: --k requires an integer of at least 2.\n' >&2
        exit 2
      fi
      k=$((10#$2))
      shift 2
      ;;
    --lambda-weight)
      if (( $# < 2 )) || [[ ! $2 =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]]; then
        printf 'Error: --lambda-weight must be a number from 0 through 1.\n' >&2
        exit 2
      fi
      lambda_weight=$2
      shift 2
      ;;
    --population)
      if (( $# < 2 )) || [[ ! $2 =~ ^[0-9]+$ ]] || (( 10#$2 < 2 )); then
        printf 'Error: --population requires an integer of at least 2.\n' >&2
        exit 2
      fi
      population=$((10#$2))
      shift 2
      ;;
    --iterations)
      if (( $# < 2 )) || [[ ! $2 =~ ^[0-9]+$ ]] || (( 10#$2 < 1 )); then
        printf 'Error: --iterations requires a positive integer.\n' >&2
        exit 2
      fi
      iterations=$((10#$2))
      shift 2
      ;;
    --seeds)
      if (( $# < 2 )) || [[ -z $2 ]]; then
        printf 'Error: --seeds requires a comma-separated list.\n' >&2
        exit 2
      fi
      seeds_text=$2
      shift 2
      ;;
    --held-out-seed)
      if (( $# < 2 )) || [[ ! $2 =~ ^[0-9]+$ ]]; then
        printf 'Error: --held-out-seed requires a non-negative integer.\n' >&2
        exit 2
      fi
      held_out_seed=$((10#$2))
      shift 2
      ;;
    --workers)
      if (( $# < 2 )) || [[ ! $2 =~ ^[0-9]+$ ]] || (( 10#$2 < 1 )); then
        printf 'Error: --workers requires a positive integer.\n' >&2
        exit 2
      fi
      workers=$((10#$2))
      shift 2
      ;;
    --worker-backend)
      if (( $# < 2 )) || [[ $2 != thread && $2 != process ]]; then
        printf 'Error: --worker-backend must be thread or process.\n' >&2
        exit 2
      fi
      worker_backend=$2
      shift 2
      ;;
    --output-directory)
      if (( $# < 2 )) || [[ -z $2 ]]; then
        printf 'Error: --output-directory requires a path.\n' >&2
        exit 2
      fi
      output_directory=$2
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
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
  usage >&2
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

maximum_clusters=$((1485 / k))
if (( cluster > maximum_clusters )); then
  printf 'Error: CLUSTER=%d is impossible for 1,485 rows with k=%d; maximum is %d.\n' \
    "$cluster" "$k" "$maximum_clusters" >&2
  exit 2
fi

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_path=${PRTI_PYTHON:-"$script_directory/.venv/bin/python"}
if [[ ! -x $python_path ]]; then
  printf 'Error: Python interpreter not found at %s. Create .venv or set PRTI_PYTHON.\n' \
    "$python_path" >&2
  exit 1
fi

if [[ -z $output_directory ]]; then
  printf -v cluster_label '%03d' "$cluster"
  lambda_label=${lambda_weight//./p}
  seeds_label=${seeds_text//,/-}
  heldout_label=$held_out_seed
  output_directory="$script_directory/evidence/prti-cluster-c${cluster_label}-k${k}-lambda${lambda_label}-pop${population}-iter${iterations}-seeds${seeds_label}-heldout${heldout_label}-v2"
fi

command=(
  "$python_path"
  "$script_directory/run_prti_experiments.py"
  dense-clusters
  --minimum-clusters "$cluster"
  --maximum-clusters "$cluster"
  --output-directory "$output_directory"
  --k "$k"
  --lambda-weight "$lambda_weight"
  --population "$population"
  --iterations "$iterations"
  --seeds "${seed_values[@]}"
  --held-out-seed "$held_out_seed"
  --workers "$workers"
  --worker-backend "$worker_backend"
  --reuse-directory "$script_directory/evidence/prti-robustness-v1"
)
if [[ $dry_run == true ]]; then
  command+=(--dry-run)
fi

printf 'PRTI-GH single-cluster study\n'
printf '  cluster count: %d\n' "$cluster"
printf '  k: %d\n' "$k"
printf '  lambda weight: %s\n' "$lambda_weight"
printf '  population: %d\n' "$population"
printf '  iterations: %d\n' "$iterations"
printf '  comparison seeds: %s\n' "$seeds_text"
printf '  held-out seed: %d\n' "$held_out_seed"
printf '  worker backend: %s\n' "$worker_backend"
printf '  output: %s\n' "$output_directory"
printf '  patient-level rows: disabled\n'
printf '  resume behavior: matching completed checkpoints are reused\n'

cd -- "$script_directory"
"${command[@]}"
