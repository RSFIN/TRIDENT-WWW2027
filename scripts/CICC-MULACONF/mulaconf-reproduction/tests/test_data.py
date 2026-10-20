from __future__ import annotations

from pathlib import Path

from mulaconf_baseline.data import load_clinc150, load_mix_dataset, stratified_partition
from mulaconf_baseline.experiment import _select_dataset_specs


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def test_supplied_dataset_sizes() -> None:
    atis = load_mix_dataset(DATA_ROOT / "MixATIS_clean", "mixatis")
    snips = load_mix_dataset(DATA_ROOT / "MixSNIPS_clean", "mixsnips")
    clinc = load_clinc150(DATA_ROOT / "CLINC150" / "data_full.json")
    assert (len(atis.train), len(atis.val), len(atis.test)) == (13162, 759, 828)
    assert (len(snips.train), len(snips.val), len(snips.test)) == (39776, 2198, 2199)
    assert (len(clinc.train), len(clinc.val), len(clinc.test), len(clinc.oos_test)) == (15000, 3000, 4500, 1000)


def test_partition_is_repeatable_and_disjoint() -> None:
    data = load_mix_dataset(DATA_ROOT / "MixSNIPS_clean", "mixsnips").val
    first, second = stratified_partition(data, 0.5, 42)
    again, _ = stratified_partition(data, 0.5, 42)
    assert first == again
    assert len(first) + len(second) == len(data)


def test_dataset_filter_selects_exactly_one_dataset() -> None:
    specs = [{"name": "mixatis"}, {"name": "mixsnips"}, {"name": "clinc150"}]
    assert _select_dataset_specs(specs, "mixsnips") == [{"name": "mixsnips"}]
    assert _select_dataset_specs(specs, None) == specs
