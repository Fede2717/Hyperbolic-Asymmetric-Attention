"""Pack tieredImageNet ImageFolder layout into per-split LMDB files.

Run on Odin after the ImageFolder symlink layout already exists at
--source-root. Writes:
  --output-root/train.lmdb/            (LMDB env directory)
  --output-root/train.lmdb.meta.json   (sidecar with classes etc.)
  --output-root/val.lmdb/
  --output-root/val.lmdb.meta.json
  --output-root/test.lmdb/
  --output-root/test.lmdb.meta.json

Class indices are taken directly from torchvision.datasets.ImageFolder
(sorted alphabetically by WNID). This matches what training has been
using until now, so the LMDB is a drop-in replacement.
"""
import argparse
import io
import json
import os
import pickle
import random
import sys

from PIL import Image


def pack_split(source_dir: str, lmdb_path: str, map_size_gb: int = 100,
               verify_n: int = 50) -> None:
    try:
        import lmdb
        from torchvision import datasets
    except ImportError as e:
        sys.exit(f"FATAL: required package missing ({e!r}). "
                 f"Install: pip install lmdb torchvision")

    if not os.path.isdir(source_dir):
        sys.exit(f"FATAL: source_dir does not exist: {source_dir}")

    print(f"[pack] reading ImageFolder at {source_dir} ...", flush=True)
    ds = datasets.ImageFolder(source_dir)
    n = len(ds.samples)
    print(f"[pack]   {n} samples, {len(ds.classes)} classes", flush=True)

    os.makedirs(os.path.dirname(lmdb_path) or ".", exist_ok=True)
    if os.path.exists(lmdb_path):
        sys.exit(f"FATAL: {lmdb_path} already exists. Remove first or use a different --output-root.")

    env = lmdb.open(lmdb_path,
                    map_size=map_size_gb * (1024 ** 3),
                    subdir=True,
                    meminit=False,
                    map_async=True,
                    writemap=True,
                    sync=False)

    print(f"[pack] writing LMDB at {lmdb_path} ...", flush=True)
    with env.begin(write=True) as txn:
        for i, (path, label) in enumerate(ds.samples):
            try:
                with open(path, "rb") as fh:
                    jpeg_bytes = fh.read()
            except (OSError, IOError) as e:
                sys.exit(f"FATAL at sample {i} ({path}): {e!r}")
            try:
                Image.open(io.BytesIO(jpeg_bytes)).verify()
            except Exception as e:
                sys.exit(f"FATAL: sample {i} at {path} is not a valid image: {e!r}")
            key = f"{i:010d}".encode("ascii")
            val = pickle.dumps({"jpeg_bytes": jpeg_bytes, "label": int(label)})
            txn.put(key, val)
            if (i + 1) % 50000 == 0:
                print(f"[pack]   wrote {i + 1} / {n}", flush=True)
    env.sync()
    env.close()
    print(f"[pack] done writing {n} samples", flush=True)

    meta = {
        "classes": ds.classes,
        "class_to_idx": ds.class_to_idx,
        "num_samples": n,
        "source_dir": os.path.abspath(source_dir),
    }
    meta_path = lmdb_path + ".meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[pack] wrote metadata at {meta_path}", flush=True)

    print(f"[verify] spot-checking {verify_n} random samples ...", flush=True)
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    rng = random.Random(0)
    indices = rng.sample(range(n), min(verify_n, n))
    mismatches = 0
    for idx in indices:
        key = f"{idx:010d}".encode("ascii")
        with env.begin(write=False) as txn:
            raw = txn.get(key)
        if raw is None:
            print(f"[verify] FAIL: key for idx {idx} not in LMDB")
            mismatches += 1
            continue
        record = pickle.loads(raw)
        src_path, src_label = ds.samples[idx]
        if int(record["label"]) != int(src_label):
            print(f"[verify] FAIL idx={idx}: lmdb label {record['label']} != src label {src_label}")
            mismatches += 1
            continue
        with open(src_path, "rb") as fh:
            src_bytes = fh.read()
        if record["jpeg_bytes"] != src_bytes:
            print(f"[verify] FAIL idx={idx} ({src_path}): byte mismatch")
            mismatches += 1
    env.close()
    if mismatches > 0:
        sys.exit(f"[verify] {mismatches} mismatches in {len(indices)} checks. LMDB is BAD.")
    print(f"[verify] {len(indices)} / {len(indices)} samples match. LMDB is OK.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", default="/media/hdd/usr/forner/tieredImageNet/")
    ap.add_argument("--output-root", default="/media/hdd/usr/forner/tieredImageNet_lmdb/")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--map-size-gb", type=int, default=100,
                    help="LMDB map_size in GB. Must exceed final on-disk size.")
    ap.add_argument("--verify-n", type=int, default=50,
                    help="How many random samples to verify per split.")
    args = ap.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    for split in args.splits:
        src = os.path.join(args.source_root, split)
        dst = os.path.join(args.output_root, f"{split}.lmdb")
        if not os.path.isdir(src):
            print(f"[pack] skipping {split} (no {src})", flush=True)
            continue
        pack_split(src, dst, map_size_gb=args.map_size_gb, verify_n=args.verify_n)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
