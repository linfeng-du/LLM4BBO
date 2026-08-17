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
python -m pip install 'pip<24.1'
pip install -e .
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn_3-3.0.0+cu128torch2.11gite2743ab-cp39-abi3-linux_x86_64.whl
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
unzip design_bench_data.zip -d $(python -c 'import site; print(site.getsitepackages()[0])')
```
