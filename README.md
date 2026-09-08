## Environment Setup

### Install MuJoCo 200

Download and extract the MuJoCo 200 binaries, then download the license key.

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

### Install Dependencies

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

### Complete the Setup

Perform runtime compilation for `mujoco_py`.

```bash
pip install 'cython<3'
python -c 'import mujoco_py'
```

Download [`design_bench_data`](https://drive.google.com/file/d/1OhhFUTiQCRb6pdyB1tqpy-qNKYbH1WFm/view?usp=sharing) and extract it into `site-packages`.

```bash
gdown 1OhhFUTiQCRb6pdyB1tqpy-qNKYbH1WFm
unzip design_bench_data.zip -d "$(python -c 'import site; print(site.getsitepackages()[0])')"
```

Locate the `trl` console script.

```bash
which trl
```

Keep the original shebang line and replace the remaining content with:

```python
# -*- coding: utf-8 -*-
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

## Download Benchmark Data

### Design-Bench TFBind10

The original [BET-seq data](https://figshare.com/articles/dataset/BET-seq_Processed_Data/5728467) include repeated measurements and model predictions for all 4^10 sequences.
Design-Bench retains multiple labels per sequence, but its lookup oracle overwrites earlier labels with later ones ([related issue](https://github.com/brandontrabucco/design-bench/issues/10)).
We instead use the Pho4 `scaled_ddG` values from `data/Manuscript_Data/scaled_nn_preds.txt.gz` in the [original data archive](https://figshare.com/ndownloader/files/10071876), giving one score per sequence.
The provided scores are already negated, so higher values indicate stronger binding.

```bash
mkdir -p src/llm4bbo/assets/design_bench/data
gdown 1DVgvac0WgywakSh0ceUW__TQhagfrbiF
unzip TFBind10_Data.zip -d src/llm4bbo/assets/design_bench/data
```

### NATS-Bench

We use the official `simple` archives, which contain precomputed evaluation results without model weights.
Oracle scores are retrieved by table lookup, so no network training is required.
Download and extract both the topology (`tss`) and size (`sss`) archives.

```bash
mkdir -p src/llm4bbo/assets/nats_bench
gdown 17_saCsj_krKjlCBLOJEpNtzPXArMCqxU
gdown 1scOMTUwcQhAMa_IMedp9lTzwmgqHLGgA
tar -xf NATS-tss-v1_0-3ffb9-simple.tar -C src/llm4bbo/assets/nats_bench
tar -xf NATS-sss-v1_0-50262-simple.tar -C src/llm4bbo/assets/nats_bench
```
