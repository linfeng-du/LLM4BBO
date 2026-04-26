#!/bin/bash

time='00-06'
tasks='tf8,tf10,ant,dkitty'

TEMP=$(getopt --options '' --longoptions 'time:,tasks:' --name "$0" -- "$@")
eval set -- "${TEMP}"

while true; do
  case "$1" in
    --time) time="$2"; shift 2 ;;
    --tasks) tasks="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

IFS=',' read -ra tasks <<< "${tasks}"

for task in "${tasks[@]}"; do
  job_name="${task}/offline_rl"
  log_dir="outputs/slurm/${job_name}"

  mkdir -p "${log_dir}"
  sbatch \
    --job-name="${job_name}" \
    --time="${time}" \
    --gres='gpu:h100:1' \
    --mem='192500M' \
    --output="${log_dir}/%j.out" \
    --error="${log_dir}/%j.err" \
    <<EOF
#!/bin/bash

source ~/.bashrc
activate llm4bbo
export OMP_NUM_THREADS=1
python -m llm4bbo.trainer.offline_rl_trainer task=${task} ${@@Q}
EOF
done
