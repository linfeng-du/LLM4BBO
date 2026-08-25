## Install MuJoCo 200

Download and extract the MuJoCo 200 binaries and the license key.

```bash
mkdir -p ~/.mujoco && cd ~/.mujoco
wget https://www.roboti.us/download/mujoco200_linux.zip
wget https://www.roboti.us/file/mjkey.txt
unzip mujoco200_linux.zip
mv mujoco200_linux mujoco200
```

Add the following lines to `~/.bashrc`, then run `source ~/.bashrc`:

```bash
if [[ ! ":$LD_LIBRARY_PATH:" =~ ":$HOME/.mujoco/mujoco200/bin:" ]]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin"
fi
```

## Environment Setup

Create a virtual environment, then install the package and its dependencies.

```bash
virtualenv --no-download llm4bbo
source llm4bbo/bin/activate
pip install 'pip<24.1'
pip install -e .
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn_3-3.0.0+cu128torch2.11gite2743ab-cp39-abi3-linux_x86_64.whl
```

Install [SOO-Bench](https://github.com/zhuyiyi-123/SOO-Bench) from source.

```bash
git clone https://github.com/zhuyiyi-123/SOO-Bench.git
cd SOO-Bench
pip install --no-deps --config-settings editable_mode=compat -e .
```

Perform runtime compilation for `mujoco_py`.

```bash
pip install 'cython<3'
python -c 'import mujoco_py'
```

Download [`design_bench_data`](https://drive.google.com/file/d/1OhhFUTiQCRb6pdyB1tqpy-qNKYbH1WFm/view?usp=sharing) and extract it into `site-packages`.

```bash
pip install gdown
gdown 1OhhFUTiQCRb6pdyB1tqpy-qNKYbH1WFm
unzip design_bench_data.zip -d "$(python -c 'import site; print(site.getsitepackages()[0])')"
```

### Patch `trl vllm-serve` to use `spawn`

Locate the `trl` console script.

```bash
which trl
```

Open the returned file and change its main content to:

```python
import multiprocessing as mp
import re
import sys
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    from trl.cli import main
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
```

This ensures that `trl vllm-serve` uses the `spawn` multiprocessing start method before importing TRL.
