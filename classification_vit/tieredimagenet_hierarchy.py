"""Load a generated tieredImageNet fine-to-superclass hierarchy.

The unfinished tieredImageNet setup utility writes ``hierarchy.json`` into
the dataset root. Hierarchy-dependent training must use that generated file;
there is deliberately no synthetic fallback because placeholder labels are
not scientifically valid training data.
"""
import json


EXPECTED_NUM_FINE = 608
EXPECTED_NUM_SUPER = 34


def load_hierarchy_from_json(path: str):
    """Return and validate the canonical tieredImageNet hierarchy JSON."""
    if not path:
        raise RuntimeError(
            "tieredImageNet hierarchy data is required for hierarchy-dependent "
            "training. Set --data_root to the generated tieredImageNet root "
            "containing hierarchy.json.")

    try:
        with open(path, "r") as fh:
            data = json.load(fh)
        num_fine = int(data["num_fine"])
        num_super = int(data["num_super"])
        raw = data["fine_to_super"]
        fine_to_super = {int(k): int(v) for k, v in raw.items()}
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError,
            json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not load a valid tieredImageNet hierarchy from '{path}'. "
            "Generate the dataset metadata with scripts/setup_tieredimagenet.py "
            "and point --data_root at that output directory.") from exc

    if num_fine != EXPECTED_NUM_FINE or num_super != EXPECTED_NUM_SUPER:
        raise RuntimeError(
            f"Unexpected tieredImageNet hierarchy dimensions in '{path}': "
            f"num_fine={num_fine}, num_super={num_super}; expected "
            f"{EXPECTED_NUM_FINE} and {EXPECTED_NUM_SUPER}.")

    expected_fine = set(range(num_fine))
    actual_fine = set(fine_to_super)
    if actual_fine != expected_fine:
        missing = len(expected_fine - actual_fine)
        extra = len(actual_fine - expected_fine)
        raise RuntimeError(
            f"Invalid fine_to_super keys in '{path}': {missing} missing and "
            f"{extra} out-of-range fine-class IDs.")

    invalid_super = sorted({s for s in fine_to_super.values()
                            if s < 0 or s >= num_super})
    if invalid_super:
        raise RuntimeError(
            f"Invalid superclass IDs in '{path}': {invalid_super}.")
    represented_super = set(fine_to_super.values())
    expected_super = set(range(num_super))
    if represented_super != expected_super:
        raise RuntimeError(
            f"Invalid fine_to_super values in '{path}': not all "
            f"{num_super} superclasses are represented.")

    return fine_to_super, num_fine, num_super
