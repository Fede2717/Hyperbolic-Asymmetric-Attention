"""Pack tieredImageNet ImageFolder layout into per-split LMDB files.

Run after the ImageFolder symlink layout already exists at
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

    # Disk-space preflight. Estimate output size from source size + 15% margin.
    import shutil
    try:
        source_bytes = 0
        for dirpath, _dirs, files in os.walk(source_dir, followlinks=True):
            for f in files:
                fp = os.path.join(dirpath, f)
                try:
                    source_bytes += os.stat(fp).st_size
                except OSError:
                    pass
    except Exception as e:
        print(f"[preflight] WARN: could not estimate source size ({e!r}); skipping space check", flush=True)
        source_bytes = 0
    if source_bytes > 0:
        required_bytes = int(source_bytes * 1.15)
        dest_parent = os.path.dirname(lmdb_path) or "."
        free_bytes = shutil.disk_usage(dest_parent).free
        print(f"[preflight] source={source_bytes / (1024**3):.2f} GB, "
              f"required (×1.15)={required_bytes / (1024**3):.2f} GB, "
              f"free at {dest_parent}={free_bytes / (1024**3):.2f} GB", flush=True)
        if free_bytes < required_bytes:
            sys.exit(f"FATAL: insufficient disk space at {dest_parent}. "
                     f"Need ~{required_bytes / (1024**3):.1f} GB, have "
                     f"{free_bytes / (1024**3):.1f} GB free. Free up space or use a different --output-root.")

    print(f"[pack] reading ImageFolder at {source_dir} ...", flush=True)
    ds = datasets.ImageFolder(source_dir)
    n = len(ds.samples)
    print(f"[pack]   {n} samples, {len(ds.classes)} classes", flush=True)

    os.makedirs(os.path.dirname(lmdb_path) or ".", exist_ok=True)
    if os.path.exists(lmdb_path):
        sys.exit(f"FATAL: {lmdb_path} already exists. Remove first or use a different --output-root.")

    import shutil as _shutil_for_rollback  # local alias to avoid name collision
    _pack_completed = False
    try:
        env = lmdb.open(lmdb_path,
                        map_size=map_size_gb * (1024 ** 3),
                        subdir=True,
                        meminit=False,
                        map_async=True,
                        writemap=True,
                        sync=False)
        try:
            print(f"[pack] writing LMDB at {lmdb_path} ...", flush=True)
            with env.begin(write=True) as txn:
                for i, (path, label) in enumerate(ds.samples):
                    try:
                        with open(path, "rb") as fh:
                            jpeg_bytes = fh.read()
                    except (OSError, IOError) as e:
                        raise RuntimeError(f"read failure at sample {i} ({path}): {e!r}")
                    try:
                        Image.open(io.BytesIO(jpeg_bytes)).verify()
                    except Exception as e:
                        raise RuntimeError(f"sample {i} at {path} is not a valid image: {e!r}")
                    key = f"{i:010d}".encode("ascii")
                    val = pickle.dumps({"jpeg_bytes": jpeg_bytes, "label": int(label)})
                    txn.put(key, val)
                    if (i + 1) % 50000 == 0:
                        print(f"[pack]   wrote {i + 1} / {n}", flush=True)
            env.sync()
        finally:
            env.close()
        _pack_completed = True
        print(f"[pack] done writing {n} samples", flush=True)
    except (KeyboardInterrupt, Exception) as e:
        if not _pack_completed and os.path.isdir(lmdb_path):
            print(f"[rollback] removing partial LMDB at {lmdb_path} due to: "
                  f"{type(e).__name__}: {e}", flush=True)
            try:
                _shutil_for_rollback.rmtree(lmdb_path, ignore_errors=False)
                # also drop any partial sidecar that may have been left
                meta_path = lmdb_path + ".meta.json"
                if os.path.exists(meta_path):
                    os.remove(meta_path)
                print(f"[rollback] cleanup successful", flush=True)
            except Exception as rb_e:
                print(f"[rollback] WARN: cleanup failed ({rb_e!r}); manual "
                      f"removal needed at {lmdb_path}", flush=True)
        raise

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
    ap.add_argument("--source-root", required=True,
                    help="Existing tieredImageNet ImageFolder root.")
    ap.add_argument("--output-root", required=True,
                    help="Destination root for per-split LMDB files.")
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
