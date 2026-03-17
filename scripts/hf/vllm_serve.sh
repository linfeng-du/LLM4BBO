#!/bin/bash

model="Qwen/Qwen3-4B"

job_name="vllm_serve/${model}"
log_dir="outputs/slurm/${job_name}"

wrap_cmds=(
    'source ~/.bashrc;'
    'activate llm4bbo;'
    'export USE_TF=0;'
    "trl vllm-serve --model ${model}"
)
wrap_cmd="${wrap_cmds[*]}"

mkdir -p "${log_dir}"
sbatch \
    --job-name="${job_name}" \
    --time='00-04' \
    --gpus-per-node='1' \
    --output="${log_dir}/%j.out" \
    --error="${log_dir}/%j.err" \
    --wrap="${wrap_cmd}"
