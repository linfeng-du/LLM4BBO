## Install `mujoco200`

Download and extract `mujoco200`.

```bash
mkdir ~/.mujoco && cd ~/.mujoco
wget https://www.roboti.us/download/mujoco200_linux.zip
wget https://www.roboti.us/file/mjkey.txt
unzip mujoco200_linux.zip && mv mujoco200_linux mujoco200
```

Add the following lines to `~/.bashrc`:

```bash
if [[ ! ":$LD_LIBRARY_PATH:" =~ ":$HOME/.mujoco/mujoco200/bin:" ]]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin"
fi
```

## Environment Setup

Create virtual environment.

```bash
virtualenv --no-download llm4bbo
source llm4bbo/bin/activate
```

Install packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

Install `design-bench` and `morphing-agents`.

```bash
pip install design-bench==2.0.20
python -m pip install 'pip<24.1' 'Cython<3'
pip install morphing-agents==1.5.1
```

Download [design_bench_data](https://drive.google.com/file/d/1OhhFUTiQCRb6pdyB1tqpy-qNKYbH1WFm/view?usp=sharing), extract into `site-packages`.

```bash
pip install gdown
gdown 1OhhFUTiQCRb6pdyB1tqpy-qNKYbH1WFm
unzip design_bench_data.zip -d llm4bbo/lib/python3.11/site-packages/
```
