from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from aic_baseline.data.manifest import resolution_scene_grouped_split
from prepare_e13_data import repeat_records_by_image_frequency


def _record(index, group, width, height, classes):
    return {
        "id": str(index),
        "group": group,
        "labels": [[class_id, 0.5, 0.5, 0.1, 0.1] for class_id in classes],
        "metadata": {"visible": {"width": width, "height": height}},
    }


def test_resolution_scene_split_is_grouped_and_stratified():
    records = []
    for index in range(80):
        records.append(
            _record(index, f"large_{index // 2}", 1920, 1080, [index % 4])
        )
    for index in range(20):
        records.append(
            _record(100 + index, f"small_{index // 2}", 640, 360, [10])
        )

    train, val, audit = resolution_scene_grouped_split(records, 0.2, 3407)
    assert len(val) == 20
    assert not ({row["group"] for row in train} & {row["group"] for row in val})
    assert audit["val"]["resolutions"]["360p_640x360"] == 4
    assert audit["val"]["resolution_scenes"]["360p_640x360|uav"] == 4


def test_oversampling_uses_class_image_frequency():
    records = [
        _record(index, f"g{index}", 1920, 1080, [0]) for index in range(10)
    ]
    records.append(_record(10, "rare", 1920, 1080, [11, 11, 11]))
    balanced, histogram = repeat_records_by_image_frequency(records, 0.5, 4)
    assert histogram[4] == 1
    assert sum(row["id"] == "10" for row in balanced) == 4


def test_generated_e13_audit_has_no_leakage():
    audit_path = Path("data/prepared/dfine_e13/audit.json")
    if not audit_path.exists():
        return
    audit = json.loads(audit_path.read_text())
    assert audit["group_overlap"] == []
    assert audit["val"]["samples"] == 400
    assert audit["val"]["resolutions"]["360p_640x360"] == 30
    assert audit["oversampling"]["tail_classes"] == [11, 1, 7, 10]


if __name__ == "__main__":
    test_resolution_scene_split_is_grouped_and_stratified()
    test_oversampling_uses_class_image_frequency()
    test_generated_e13_audit_has_no_leakage()
    print("E13 data tests passed")
