from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mulaconf_baseline.data import load_clinc150, load_mix_dataset  # noqa: E402


def main() -> None:
    datasets = [
        load_mix_dataset(PROJECT_ROOT / "../data/MixATIS_clean", "mixatis"),
        load_mix_dataset(PROJECT_ROOT / "../data/MixSNIPS_clean", "mixsnips"),
        load_clinc150(PROJECT_ROOT / "../data/CLINC150/data_full.json"),
    ]
    for dataset in datasets:
        labels = set().union(*(set(row.labels) for row in dataset.train + dataset.val + dataset.test if not row.is_oos))
        cardinalities = Counter(len(row.labels) for row in dataset.train)
        print(
            f"{dataset.name}: train={len(dataset.train)}, val={len(dataset.val)}, test={len(dataset.test)}, "
            f"oos_val={len(dataset.oos_val)}, oos_test={len(dataset.oos_test)}, "
            f"atomic_labels={len(labels)}, train_cardinality={dict(sorted(cardinalities.items()))}"
        )


if __name__ == "__main__":
    main()
