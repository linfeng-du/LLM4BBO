#!/bin/bash

tasks='tf8,tf10,ant,dkitty'

TEMP=$(getopt --options '' --longoptions 'tasks:' --name "$0" -- "$@")
eval set -- "${TEMP}"

while true; do
  case "$1" in
    --tasks) tasks="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

IFS=',' read -ra tasks <<< "${tasks}"

for task in "${tasks[@]}"; do
  job_name="${task}/online_rl"
  log_dir="outputs/slurm/${job_name}"

  wrap_cmds=(
      'source ~/.bashrc;'
      'activate llm4bbo;'
      "OMP_NUM_THREADS=1 python src/train_online_rl.py task=${task};"
      "python src/test.py task=${task} stage=online_rl"
  )
  wrap_cmd="${wrap_cmds[*]}"

  mkdir -p "${log_dir}"
  sbatch \
      --job-name="${job_name}" \
      --time='00-12' \
      --gpus-per-node='1' \
      --output="${log_dir}/%j.out" \
      --error="${log_dir}/%j.err" \
      --wrap="${wrap_cmd}"
done
