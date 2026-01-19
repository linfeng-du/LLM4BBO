#!/bin/bash

sbatch_time='12:00:00'
task_names="TFBind8-Exact-v0,TFBind10-Exact-v0,\
AntMorphology-Exact-v0,DKittyMorphology-Exact-v0"
dataset_sizes='900,700,500,300'
llms='qwen3-4b-instruct'

LONGOPTIONS='time:,task_names:,dataset_sizes:,llms:'
TEMP=$(getopt --options '' --longoptions "${LONGOPTIONS}" --name "$0" -- "$@")
eval set -- "${TEMP}"

while true; do
  case "$1" in
    --time) sbatch_time="$2"; shift 2 ;;
    --task_names) task_names="$2"; shift 2 ;;
    --dataset_sizes) dataset_sizes="$2"; shift 2 ;;
    --llms) llms="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

IFS=',' read -r -a task_names <<< "${task_names}"
IFS=',' read -r -a dataset_sizes <<< "${dataset_sizes}"
IFS=',' read -r -a llms <<< "${llms}"

for llm in "${llms[@]}"; do
  for task_name in "${task_names[@]}"; do
    for dataset_size in "${dataset_sizes[@]}"; do
        job_name="${llm}/${task_name}/d${dataset_size}/train_offline"
        log_dir="outputs/slurm/${job_name}"

        wrap_cmds=(
            'source ~/.bashrc;'
            'activate llm4bbo;'
            'python src/train_offline.py'
            "task_name=${task_name}"
            "dataset_size=${dataset_size}"
            "llm=${llm}"
        )
        wrap_cmd="${wrap_cmds[*]}"

        mkdir -p "${log_dir}"
        sbatch \
            --job-name="${job_name}" \
            --time="${sbatch_time}" \
            --gpus-per-node=1 \
            --output="${log_dir}/%j.out" \
            --error="${log_dir}/%j.err" \
            --wrap="${wrap_cmd}"
    done
  done
done
