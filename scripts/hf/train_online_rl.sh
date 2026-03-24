#!/bin/bash

time='01-00'
tasks='tf8,tf10,ant,dkitty'

LONGOPTIONS='time:,tasks:,host:'
TEMP=$(getopt --options '' --longoptions "${LONGOPTIONS}" --name "$0" -- "$@")
eval set -- "${TEMP}"

while true; do
  case "$1" in
    --time) time="$2"; shift 2 ;;
    --tasks) tasks="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
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
    'python -m llm4bbo.trainer.hf.online_rl_trainer'
    "task=${task}"
    "grpo_config.vllm_server_host=${host}"
  )
  wrap_cmd="${wrap_cmds[*]}"

  mkdir -p "${log_dir}"
  sbatch \
    --job-name="${job_name}" \
    --time="${time}" \
    --gpus-per-node='1' \
    --exclude="${host}" \
    --output="${log_dir}/%j.out" \
    --error="${log_dir}/%j.err" \
    --wrap="${wrap_cmd}"
done
