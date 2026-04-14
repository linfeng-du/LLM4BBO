#!/bin/bash

time='01-00'
tasks='tf8,tf10,ant,dkitty'
host=''
port=''

LONGOPTIONS='time:,tasks:,host:,port:'
TEMP=$(getopt --options '' --longoptions "${LONGOPTIONS}" --name "$0" -- "$@")
eval set -- "${TEMP}"

while true; do
  case "$1" in
    --time) time="$2"; shift 2 ;;
    --tasks) tasks="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${host}" || -z "${port}" ]]; then
  echo "Missing --host or --port" >&2
  exit 1
fi

IFS=',' read -ra tasks <<< "${tasks}"
hydra_overrides=("$@")

for task in "${tasks[@]}"; do
  job_name="${task}/online_rl"
  log_dir="outputs/slurm/${job_name}"

  wrap_cmds=(
    'source ~/.bashrc;'
    'activate llm4bbo;'
    'python -m llm4bbo.trainer.online_rl_trainer'
    "task=${task}"
    "grpo_config.vllm_server_host=${host}"
    "grpo_config.vllm_server_port=${port}"
    "${hydra_overrides[@]}"
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
