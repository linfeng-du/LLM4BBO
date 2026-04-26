#!/bin/bash

time='01-00'
task=''
host=''
port=''

LONGOPTIONS='time:,task:,host:,port:'
TEMP=$(getopt --options '' --longoptions "${LONGOPTIONS}" --name "$0" -- "$@")
eval set -- "${TEMP}"

while true; do
  case "$1" in
    --time) time="$2"; shift 2 ;;
    --task) task="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${task}" || -z "${host}" || -z "${port}" ]]; then
  echo "Missing --task, --host or --port" >&2
  exit 1
fi

job_name="${task}/online_rl_server"
log_dir="outputs/slurm/${job_name}"

mkdir -p "${log_dir}"
sbatch \
  --job-name="${job_name}" \
  --time="${time}" \
  --gres='gpu:h100:1' \
  --mem='192500M' \
  --exclude="${host}" \
  --output="${log_dir}/%j.out" \
  --error="${log_dir}/%j.err" \
  <<EOF
#!/bin/bash

source ~/.bashrc
activate llm4bbo
python -m llm4bbo.trainer.online_rl_trainer \
  task=${task} \
  grpo_config.vllm_mode=server \
  +grpo_config.vllm_server_host=${host} \
  +grpo_config.vllm_server_port=${port} \
  ${@@Q}
EOF
