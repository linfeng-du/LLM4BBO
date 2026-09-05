import collections
import logging
import os
import sys
import warnings
from collections.abc import Generator
from contextlib import contextmanager

from packaging.version import Version


def _silence_noisy_output() -> None:
    logging.getLogger().addFilter(_RobelMujocoFilter())
    warnings.filterwarnings("ignore", module=r"gym")
    warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated")


class _RobelMujocoFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(("[-0.  1.]", "MuJoCo"))


def _patch_collections_mapping() -> None:
    if sys.version_info >= (3, 10):
        collections.Mapping = collections.abc.Mapping


def _patch_numpy_inf() -> None:
    import numpy as np

    if Version(np.__version__) >= Version("2.0"):
        np.NINF = -np.inf
        np.PINF = np.inf


def _patch_smiles_tokenizer_init() -> None:
    with _silence_stdout_stderr():
        from deepchem.feat.smiles_tokenizer import (
            BasicSmilesTokenizer,
            SmilesTokenizer,
            load_vocab
        )

    # https://github.com/deepchem/deepchem/blob/2.8.0/deepchem/feat/smiles_tokenizer.py#L68-L99
    def __init__(
        self,
        vocab_file: str = '',
        # unk_token="[UNK]",
        # sep_token="[SEP]",
        # pad_token="[PAD]",
        # cls_token="[CLS]",
        # mask_token="[MASK]",
        **kwargs):
        """Constructs a SmilesTokenizer.

        Parameters
        ----------
        vocab_file: str
            Path to a SMILES character per line vocabulary file.
            Default vocab file is found in deepchem/feat/tests/data/vocab.txt
        """

        super(SmilesTokenizer, self).__init__(vocab_file, **kwargs)

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocab file at path '{}'.".format(vocab_file))
        self._smiles_vocab = load_vocab(vocab_file)
        self.highest_unused_index = max([
            i for i, v in enumerate(self._smiles_vocab.keys())
            if v.startswith("[unused")
        ])
        self.ids_to_tokens = collections.OrderedDict([
            (ids, tok) for tok, ids in self._smiles_vocab.items()
        ])
        self.basic_tokenizer = BasicSmilesTokenizer()

    SmilesTokenizer.__init__ = __init__
    SmilesTokenizer.vocab = property(lambda self: self._smiles_vocab)


@contextmanager
def _silence_stdout_stderr() -> Generator[None, None, None]:
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stdout_fd = os.dup(sys.stdout.fileno())
    stderr_fd = os.dup(sys.stderr.fileno())

    try:
        os.dup2(devnull_fd, sys.stdout.fileno())
        os.dup2(devnull_fd, sys.stderr.fileno())
        yield

    finally:
        os.dup2(stdout_fd, sys.stdout.fileno())
        os.dup2(stderr_fd, sys.stderr.fileno())

        os.close(devnull_fd)
        os.close(stdout_fd)
        os.close(stderr_fd)


def _patch_dkitty_env_init() -> None:
    from morphing_agents.mujoco.dkitty.env import DKittyEnv

    original_init = DKittyEnv.__init__

    def __init__(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)

        # Upstream writes XML to a file, then RobotEnv loads it to create the model
        # The following blocks write incorrect XML ranges using hip_range instead of thigh_range:
        # https://github.com/brandontrabucco/morphing-agents/blob/master/morphing_agents/mujoco/dkitty/env.py#L130-L137
        # https://github.com/brandontrabucco/morphing-agents/blob/master/morphing_agents/mujoco/dkitty/env.py#L161-L168
        # https://github.com/brandontrabucco/morphing-agents/blob/master/morphing_agents/mujoco/dkitty/env.py#L192-L199
        # https://github.com/brandontrabucco/morphing-agents/blob/master/morphing_agents/mujoco/dkitty/env.py#L223-L230
        thigh_names = ("A:FRJ11", "A:FLJ21", "A:BLJ31", "A:BRJ41")

        for leg, thigh_name in zip(self._legs, thigh_names):
            thigh_range = (
                leg.thigh_center - leg.thigh_range,
                leg.thigh_center + leg.thigh_range
            )

            joint_id = self.model.joint_name2id(thigh_name)
            actuator_id = self.model.actuator_name2id(thigh_name)
            self.model.jnt_range[joint_id] = thigh_range
            self.model.actuator_ctrlrange[actuator_id] = thigh_range

    DKittyEnv.__init__ = __init__


_silence_noisy_output()
_patch_collections_mapping()
_patch_numpy_inf()
_patch_smiles_tokenizer_init()
_patch_dkitty_env_init()
