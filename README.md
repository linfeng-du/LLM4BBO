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

Set up `mujoco200`.

```bash
cd ~ && mkdir .mujoco && cd .mujoco
wget https://www.roboti.us/download/mujoco200_linux.zip
wget https://www.roboti.us/file/mjkey.txt
unzip mujoco200_linux.zip && mv mujoco200_linux mujoco200
```

Add the following line to `~/.bashrc`:

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin"
```

Install `design-bench` and `morphing-agents`.

```bash
pip install design-bench==2.0.20

python -m pip install 'pip<24.1'
pip install 'Cython<3'
pip install morphing-agents==1.5.1
```

Download [design_bench_data](https://drive.google.com/file/d/1OhhFUTiQCRb6pdyB1tqpy-qNKYbH1WFm/view?usp=sharing), unzip into `site-packages`.

```bash
pip install gdown
gdown 1OhhFUTiQCRb6pdyB1tqpy-qNKYbH1WFm
unzip design_bench_data.zip -d llm4bbo/lib/python3.11/site-packages/
```
