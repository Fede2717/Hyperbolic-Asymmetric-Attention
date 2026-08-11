"""Dataset-agnostic hierarchy loader. Picks the right fine->super mapping
module based on the dataset name."""
def load_hierarchy(dataset_name: str, hierarchy_path: str = None):
    if dataset_name == 'CIFAR-100':
        try:
            from classification_vit.cifar100_hierarchy import (
                FINE_TO_SUPER, NUM_FINE, NUM_SUPER,
            )
        except ImportError:
            from cifar100_hierarchy import (
                FINE_TO_SUPER, NUM_FINE, NUM_SUPER,
            )
    elif dataset_name == 'tieredImageNet':
        try:
            from classification_vit.tieredimagenet_hierarchy import load_hierarchy_from_json
        except ImportError:
            from tieredimagenet_hierarchy import load_hierarchy_from_json
        return load_hierarchy_from_json(hierarchy_path)
    else:
        raise NotImplementedError(
            f"No hierarchy module available for dataset '{dataset_name}'. "
            f"Add one or run with hierarchy-dependent losses disabled.")
    return FINE_TO_SUPER, NUM_FINE, NUM_SUPER
