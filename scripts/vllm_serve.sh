#!/bin/bash

time='01-00'
model="Qwen/Qwen3-4B"
port=''

LONGOPTIONS='time:,model:,port:'
TEMP=$(getopt --options '' --longoptions "${LONGOPTIONS}" --name "$0" -- "$@")
eval set -- "${TEMP}"

while true; do
  case "$1" in
    --time) time="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${port}" ]]; then
  echo "Missing --port" >&2
  exit 1
fi

job_name="vllm_serve/${model}"
log_dir="outputs/slurm/${job_name}"

wrap_cmds=(
    'source ~/.bashrc;'
    'activate llm4bbo;'
    'export USE_TF=0;'
    "trl vllm-serve --model ${model} --port ${port}"
)
wrap_cmd="${wrap_cmds[*]}"

mkdir -p "${log_dir}"
sbatch \
  --job-name="${job_name}" \
  --time="${time}" \
  --gres='gpu:h100:1' \
  --mem='192500M' \
  --output="${log_dir}/%j.out" \
  --error="${log_dir}/%j.err" \
  --wrap="${wrap_cmd}"
