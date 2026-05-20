"""LMDB-backed dataset that mimics torchvision.datasets.ImageFolder.

Designed for tieredImageNet to escape network-filesystem I/O when the
JPEGs live on a slow shared drive. Stores raw JPEG bytes per sample
keyed by integer index. Class indices match what ImageFolder would
produce (sorted by WNID alphabetically — same as ImageFolder default).

LMDB environments are opened LAZILY inside __getitem__ to be safe
across forked DataLoader workers — never open in __init__.
"""
import io
import json
import os
import pickle
from typing import Optional, Callable, Tuple, Any

from PIL import Image
from torch.utils.data import Dataset

try:
    import lmdb
except ImportError as e:
    raise ImportError(
        "The lmdb package is required for LMDBImageFolder. "
        "Install with: pip install lmdb"
    ) from e


class LMDBImageFolder(Dataset):
    """Drop-in replacement for datasets.ImageFolder backed by LMDB.

    Expects two files at `lmdb_path`:
      - <lmdb_path>           : the LMDB environment directory
      - <lmdb_path>.meta.json : metadata sidecar with classes, class_to_idx, num_samples

    Returns the same (PIL.Image, int_label) tuple ImageFolder would
    before transform is applied, so the existing transform pipelines
    work without modification.
    """

    def __init__(self, lmdb_path: str, transform: Optional[Callable] = None):
        self.lmdb_path = os.path.abspath(lmdb_path)
        self.transform = transform

        meta_path = self.lmdb_path + ".meta.json"
        if not os.path.isdir(self.lmdb_path):
            raise FileNotFoundError(
                f"LMDB directory does not exist: {self.lmdb_path}"
            )
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"LMDB metadata sidecar does not exist: {meta_path}"
            )

        with open(meta_path) as fh:
            meta = json.load(fh)
        self.classes = meta["classes"]
        self.class_to_idx = meta["class_to_idx"]
        self.num_samples = int(meta["num_samples"])

        assert len(self.classes) == len(self.class_to_idx), \
            f"classes/class_to_idx length mismatch in {meta_path}"
        assert set(self.class_to_idx.values()) == set(range(len(self.classes))), \
            f"class_to_idx values are not 0..N-1 in {meta_path}"

        self._env = None

    def _ensure_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=512,
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"index {index} out of range [0, {self.num_samples})")
        self._ensure_env()
        key = f"{index:010d}".encode("ascii")
        with self._env.begin(write=False) as txn:
            raw = txn.get(key)
        if raw is None:
            raise KeyError(f"LMDB missing key for index {index}")
        record = pickle.loads(raw)
        jpeg_bytes = record["jpeg_bytes"]
        label = int(record["label"])
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._env = None


def __repr_for_debug(self):
    return (f"LMDBImageFolder(lmdb_path={self.lmdb_path!r}, "
            f"num_samples={self.num_samples}, "
            f"num_classes={len(self.classes)})")
LMDBImageFolder.__repr__ = __repr_for_debug
