import logging
import os
import sys
import warnings
from collections.abc import Generator
from contextlib import contextmanager


class WarningErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        print(record.getMessage())
        return not record.getMessage().startswith(("[-0.  1.]", "MuJoCo"))


logging.getLogger().addFilter(WarningErrorFilter())
warnings.filterwarnings("ignore", module=r"gym")


@contextmanager
def silence_stdout_stderr() -> Generator[None, None, None]:
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


with silence_stdout_stderr():
    from packaging.version import Version

    import deepchem as dc
    import numpy as np


logger = logging.getLogger(__name__)


if sys.version_info >= (3, 10):
    import collections

    collections.Mapping = collections.abc.Mapping
    logger.info("Patched collections.Mapping")


if Version(np.__version__) >= Version("2.0"):
    np.NINF = -np.inf
    np.PINF = np.inf
    logger.info("Patched numpy.NINF and numpy.PINF")


if Version(dc.__version__) == Version("2.5.0"):
    import collections
    import os

    from deepchem.feat.smiles_tokenizer import (
        BasicSmilesTokenizer,
        SmilesTokenizer,
        load_vocab
    )

    # https://github.com/deepchem/deepchem/blob/2.8.0/deepchem/feat/smiles_tokenizer.py#L68
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
        self.vocab = load_vocab(vocab_file)
        self.highest_unused_index = max([
            i for i, v in enumerate(self.vocab.keys())
            if v.startswith("[unused")
        ])
        self.ids_to_tokens = collections.OrderedDict([
            (ids, tok) for tok, ids in self.vocab.items()
        ])
        self.basic_tokenizer = BasicSmilesTokenizer()

    SmilesTokenizer.__init__ = __init__
    logger.info("Patched deepchem.feat.smiles_tokenizer.SmilesTokenizer")
