"""tieredImageNet builder.

Reads scripts/tieredimagenet_splits.txt (608 fine -> super pairs from
Ren et al. 2018, see file header), verifies that every fine WNID has
an ILSVRC-12 train folder under --imagenet-root, then creates a
symlinked ImageFolder layout at --output-root and writes
hierarchy.json + class_index.json there.

Symlinks (not copies) keep the new dataset ~MB instead of ~80GB.
"""
import argparse, json, os, sys
from datetime import date

SPLITS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "tieredimagenet_splits.txt")
SOURCE_URL = ("https://github.com/renmengye/few-shot-ssl-public/tree/"
              "master/fewshot/data/tiered_imagenet_split")

def load_splits():
    fine_to_super = {}
    with open(SPLITS_FILE) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fine, sup = line.split()
            fine_to_super[fine] = sup
    fine_wnids = sorted(fine_to_super.keys())
    super_wnids = sorted(set(fine_to_super.values()))
    assert len(fine_wnids) == 608, len(fine_wnids)
    assert len(super_wnids) == 34, len(super_wnids)
    return fine_wnids, super_wnids, fine_to_super

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenet-root", required=True,
                    help="Existing ILSVRC-12 training root containing WNID directories.")
    ap.add_argument("--output-root", required=True,
                    help="Destination for the generated tieredImageNet layout and metadata.")
    ap.add_argument("--use-val-as-test", action="store_true",
                    help="Mirror val/ to test/ (Tiny-ImageNet convention).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force",   action="store_true",
                    help="Remove existing symlinks before re-creating.")
    args = ap.parse_args()

    if sys.version_info < (3, 8):
        sys.exit("Python 3.8+ required.")

    fine_wnids, super_wnids, fine_to_super = load_splits()

    # 1. ImageNet sanity
    for sub in ("train", "val"):
        p = os.path.join(args.imagenet_root, sub)
        if not os.path.isdir(p):
            sys.exit(f"FATAL: {p} does not exist or is not a directory.")
    missing = [w for w in fine_wnids
               if not os.path.isdir(os.path.join(args.imagenet_root, "train", w))]
    if missing:
        print(f"FATAL: {len(missing)} fine WNIDs missing under "
              f"{args.imagenet_root}/train. First 5: {missing[:5]}",
              file=sys.stderr)
        sys.exit(1)

    # 2. WordNet sanity + cross-check (lazy import).
    try:
        from nltk.corpus import wordnet as wn
    except Exception as e:
        sys.exit(f"FATAL: nltk.corpus.wordnet not importable ({e!r}). "
                 f"Run: pip install nltk && python -c \"import nltk; nltk.download('wordnet')\"")
    def syn(wnid):
        return wn.synset_from_pos_and_offset("n", int(wnid[1:]))
    for w in super_wnids + fine_wnids:
        try:
            syn(w)
        except Exception as e:
            sys.exit(f"FATAL: WordNet lookup failed for {w}: {e!r}")

    # 3. Integer indices.
    fine_to_idx  = {w: i for i, w in enumerate(fine_wnids)}
    super_to_idx = {w: i for i, w in enumerate(super_wnids)}
    fine_to_super_idx = {fine_to_idx[w]: super_to_idx[fine_to_super[w]]
                         for w in fine_wnids}
    assert len(fine_to_super_idx) == 608
    assert set(fine_to_super_idx.values()) == set(range(34))

    # 4. WordNet hypernym cross-check (warn-only).
    disagreements = []
    for w in fine_wnids:
        s = syn(w)
        target = syn(fine_to_super[w])
        ancestors = {x for path in s.hypernym_paths() for x in path}
        if target not in ancestors:
            disagreements.append((w, fine_to_super[w]))

    os.makedirs(args.output_root, exist_ok=True)
    if disagreements:
        disagree_path = os.path.join(args.output_root,
                                     "tieredimagenet_wordnet_disagreements.txt")
        if not args.dry_run:
            with open(disagree_path, "w") as fh:
                for w, sup in disagreements:
                    fh.write(f"{w} {sup}\n")
        print(f"WARN: {len(disagreements)} fine WNIDs whose declared super "
              f"is not a WordNet hypernym ancestor (written to "
              f"{disagree_path}). Not aborting.", file=sys.stderr)

    # 5. Symlink construction.
    totals = {"train": 0, "val": 0, "test": 0}
    for split_src, split_dst in [("train", "train"), ("val", "val")]:
        for w in fine_wnids:
            src_dir = os.path.join(args.imagenet_root, split_src, w)
            dst_dir = os.path.join(args.output_root, split_dst, w)
            if not args.dry_run:
                os.makedirs(dst_dir, exist_ok=True)
            files = sorted(os.listdir(src_dir)) if os.path.isdir(src_dir) else []
            first_three_logged = 0
            for fn in files:
                src = os.path.join(src_dir, fn)
                dst = os.path.join(dst_dir, fn)
                if args.dry_run:
                    if first_three_logged < 3:
                        print(f"DRY-RUN ln -s {src} -> {dst}")
                        first_three_logged += 1
                else:
                    if os.path.lexists(dst):
                        if args.force:
                            os.remove(dst)
                        else:
                            totals[split_dst] += 1
                            continue
                    os.symlink(src, dst)
                totals[split_dst] += 1

    if args.use_val_as_test:
        for w in fine_wnids:
            src_dir = os.path.join(args.imagenet_root, "val", w)
            dst_dir = os.path.join(args.output_root, "test", w)
            if not args.dry_run:
                os.makedirs(dst_dir, exist_ok=True)
            files = sorted(os.listdir(src_dir)) if os.path.isdir(src_dir) else []
            first_three_logged = 0
            for fn in files:
                src = os.path.join(src_dir, fn)
                dst = os.path.join(dst_dir, fn)
                if args.dry_run:
                    if first_three_logged < 3:
                        print(f"DRY-RUN ln -s {src} -> {dst}")
                        first_three_logged += 1
                else:
                    if os.path.lexists(dst):
                        if args.force:
                            os.remove(dst)
                        else:
                            totals["test"] += 1
                            continue
                    os.symlink(src, dst)
                totals["test"] += 1

    # 6. JSON outputs.
    hierarchy = {
        "num_fine": 608,
        "num_super": 34,
        "fine_to_super": {str(k): v for k, v in fine_to_super_idx.items()},
        "_meta": {
            "source_url": SOURCE_URL,
            "fetched_on": "2026-05-15",
            "tool": "scripts/setup_tieredimagenet.py",
            "generated_on": date.today().isoformat(),
        },
    }
    class_index = {
        "wnid_to_fine_idx": fine_to_idx,
        "fine_idx_to_wnid": fine_wnids,
        "wnid_to_super_idx": super_to_idx,
        "super_idx_to_wnid": super_wnids,
    }
    h_path = os.path.join(args.output_root, "hierarchy.json")
    c_path = os.path.join(args.output_root, "class_index.json")
    if not args.dry_run:
        with open(h_path, "w") as fh:
            json.dump(hierarchy, fh, indent=2)
        with open(c_path, "w") as fh:
            json.dump(class_index, fh, indent=2)

    # 7. Summary.
    print(f"OK fine={len(fine_wnids)} super={len(super_wnids)} "
          f"symlinks train={totals['train']} val={totals['val']} "
          f"test={totals['test']} "
          f"hierarchy_json={h_path} disagreements={len(disagreements)}")

if __name__ == "__main__":
    main()
