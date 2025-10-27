## Environment Setup (the Alliance)
Load `mujoco` and its dependencies.
It is recommended to add the following commands to `~/.bashrc`.

```bash
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.6
module load arrow/21.0.0
module load rust/1.85.0
module load python/3.11.5
module load mujoco/3.3.0
```

Setup virtual environment.

```bash
virtualenv --no-download llm4bbo
source llm4bbo/bin/activate
```

Install `robel` from source (required by `morphing-agents`).

```bash
git clone --recurse-submodules https://github.com/google-research/robel.git
sed -i 's/transforms3d>=0\.3\.0<0\.4/transforms3d>=0.3.0,<0.4/' robel/requirements.txt  # Fix typo
pip install -e robel/
```

Install packages from `requirements.txt`.

```bash 
pip install -r requirements.txt
```

Download [design_bench_data](https://drive.google.com/file/d/1OhhFUTiQCRb6pdyB1tqpy-qNKYbH1WFm/view?usp=sharing),
unzip it into `site-packages`.

```bash
unzip design_bench_data.zip -d llm4bbo/lib/python3.11/site-packages/
```

### Resolve Version Incompatibilities

#### `import design_bench`

`llm4bbo/lib/python3.11/site-packages/deepchem/feat/smiles_tokenizer.py`

- Comment `self.max_len_single_sentence = self.max_len - 2` and `self.max_len_sentences_pair = self.max_len - 3`.

- Replace `self.max_len` with `self.model_max_length`.

#### `design_bench.make('TFBind8-Exact-v0')`

`llm4bbo/lib/python3.11/site-packages/design_bench/datasets/dataset_builder.py`

- Replace `np.bool` with `np.bool_`.

`llm4bbo/lib/python3.11/site-packages/gym/envs/mujoco/mujoco_env.py`

- Replace `import mujoco_py` with `import mujoco`.

`llm4bbo/lib/python3.11/site-packages/gym/envs/mujoco/pusher.py`

- Comment `import mujoco_py`.

`llm4bbo/lib/python3.11/site-packages/transforms3d/quaternions.py`

- Replace `np.float` with `np.float64`.

#### `design_bench.make('GFP-Transformer-v0')`

`llm4bbo/lib/python3.11/site-packages/design_bench/oracles/approximate_oracle.py`

- Replace `rank_correlation = np.loads(file.read())` with `rank_correlation = np.load(file, allow_pickle=True)`.

#### `design_bench.make('Superconductor-RandomForest-v0')`

`llm4bbo/lib/python3.11/site-packages/design_bench/oracles/approximate_oracle.py`

```python
import sklearn.tree._tree as _tree

node_dtype = _tree.NODE_DTYPE
_tree.NODE_DTYPE = np.dtype([
    ("left_child", "<i8"),
    ("right_child", "<i8"),
    ("feature", "<i8"),
    ("threshold", "<f8"),
    ("impurity", "<f8"),
    ("n_node_samples", "<i8"),
    ("weighted_n_node_samples", "<f8"),
])
self.params = self.load_params(self.resource.disk_target)  # ApproximateOracle.__init__
_tree.NODE_DTYPE = node_dtype
```
