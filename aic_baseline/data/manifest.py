from __future__ import annotations

import collections
import json
import math
import random
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from aic_baseline.constants import MODALITIES, NUM_CLASSES


@dataclass(frozen=True)
class ImageMetadata:
    format: str
    width: int
    height: int
    bit_depth: int
    channels: int


def _image_metadata(handle) -> ImageMetadata:
    data = handle.read(131072)
    if data[:8] == bytes.fromhex("89504e470d0a1a0a"):
        width, height, bit_depth, color_type = struct.unpack(">IIBB", data[16:26])
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
        return ImageMetadata("png", width, height, bit_depth, channels)
    if data[:2] == bytes.fromhex("ffd8"):
        index = 2
        sof_markers = set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0))
        while index + 9 <= len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            while index < len(data) and data[index] == 0xFF:
                index += 1
            marker = data[index]
            index += 1
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                continue
            segment_length = int.from_bytes(data[index:index + 2], "big")
            if marker in sof_markers:
                bit_depth = data[index + 2]
                height = int.from_bytes(data[index + 3:index + 5], "big")
                width = int.from_bytes(data[index + 5:index + 7], "big")
                channels = data[index + 7]
                return ImageMetadata("jpg", width, height, bit_depth, channels)
            if segment_length < 2:
                break
            index += segment_length
    raise ValueError("Unsupported or malformed image")


def derive_group_id(stem: str, suffix: str) -> str:
    """Create a conservative sequence group to reduce adjacent-frame leakage."""
    parts = stem.split("_")
    if len(parts) >= 5 and parts[0].lower() == "hehe" and parts[1].isdigit():
        return "seq_" + "_".join(parts[2:4])
    if len(parts) >= 2:
        return "seq_" + "_".join(parts[:2])
    if stem.isdigit():
        value = int(stem)
        # Numeric JPG frames are strongly sequential. PNG numeric identifiers are
        # grouped more tightly because they may be independent captures.
        bin_size = 25 if suffix.lower() in {".jpg", ".jpeg"} else 10
        return f"numeric_{suffix.lower().lstrip('.')}_{value // bin_size:06d}"
    return "misc_" + stem[:24]


def _parse_labels(raw: bytes, member: str) -> tuple[list[list[float]], list[dict]]:
    labels: list[list[float]] = []
    issues: list[dict] = []
    text = raw.decode("utf-8-sig", errors="strict").strip()
    if not text:
        return labels, issues
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{member}:{line_number}: expected 5 fields, got {len(fields)}")
        class_id = int(fields[0])
        x, y, width, height = map(float, fields[1:])
        if not 0 <= class_id < NUM_CLASSES:
            raise ValueError(f"{member}:{line_number}: invalid class {class_id}")
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError(f"{member}:{line_number}: non-finite coordinates")
        if not (width > 0 and height > 0):
            raise ValueError(f"{member}:{line_number}: invalid normalized box size")
        overflow = max(0.0, width / 2 - x, x + width / 2 - 1, height / 2 - y, y + height / 2 - 1)
        if overflow > 1e-6:
            issues.append({"member": member, "line": line_number, "overflow": overflow})
        x1 = min(max(x - width / 2, 0.0), 1.0)
        y1 = min(max(y - height / 2, 0.0), 1.0)
        x2 = min(max(x + width / 2, 0.0), 1.0)
        y2 = min(max(y + height / 2, 0.0), 1.0)
        if x2 <= x1 or y2 <= y1:
            issues.append({"member": member, "line": line_number, "overflow": overflow, "dropped": True})
            continue
        labels.append([float(class_id), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
    return labels, issues


def scan_archive(archive: Path, data_root: Path, require_labels: bool) -> tuple[list[dict], dict]:
    records: list[dict] = []
    issues: list[dict] = []
    with zipfile.ZipFile(archive) as zipped:
        members: dict[str, dict[str, str]] = collections.defaultdict(dict)
        for info in zipped.infolist():
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename)
            members[path.parent.name][path.stem] = info.filename

        image_sets = {modality: set(members[modality]) for modality in MODALITIES}
        if len({frozenset(values) for values in image_sets.values()}) != 1:
            details = {key: len(value) for key, value in image_sets.items()}
            raise ValueError(f"Modalities are not one-to-one in {archive}: {details}")
        stems = sorted(image_sets["visible"])
        if require_labels and set(members["labels"]) != set(stems):
            raise ValueError(f"Labels are not one-to-one with images in {archive}")

        signatures: dict[str, collections.Counter] = {modality: collections.Counter() for modality in MODALITIES}
        class_counts = collections.Counter()
        image_class_counts = collections.Counter()
        for stem in stems:
            image_members = {modality: members[modality][stem] for modality in MODALITIES}
            metadata = {}
            for modality, member in image_members.items():
                with zipped.open(member) as handle:
                    current = _image_metadata(handle)
                metadata[modality] = current.__dict__
                signatures[modality][tuple(current.__dict__.values())] += 1
            dimensions = {(value["width"], value["height"]) for value in metadata.values()}
            if len(dimensions) != 1:
                raise ValueError(f"Cross-modal dimension mismatch for {stem}: {metadata}")

            labels: list[list[float]] = []
            if require_labels:
                label_member = members["labels"][stem]
                labels, label_issues = _parse_labels(zipped.read(label_member), label_member)
                issues.extend(label_issues)
                classes_in_image = set()
                for label in labels:
                    class_id = int(label[0])
                    class_counts[class_id] += 1
                    classes_in_image.add(class_id)
                image_class_counts.update(classes_in_image)

            visible_suffix = PurePosixPath(image_members["visible"]).suffix
            records.append(
                {
                    "id": stem,
                    "archive": str(archive.relative_to(data_root)),
                    "members": image_members,
                    "label_member": members["labels"].get(stem),
                    "labels": labels,
                    "metadata": metadata,
                    "group": derive_group_id(stem, visible_suffix),
                }
            )

    summary = {
        "archive": str(archive),
        "samples": len(records),
        "boxes": sum(class_counts.values()),
        "class_counts": {str(key): class_counts[key] for key in range(NUM_CLASSES)},
        "class_image_counts": {str(key): image_class_counts[key] for key in range(NUM_CLASSES)},
        "boundary_overflow_boxes": len(issues),
        "boundary_issue_examples": sorted(issues, key=lambda item: item["overflow"], reverse=True)[:20],
        "signatures": {
            modality: [
                {"count": count, "format": key[0], "width": key[1], "height": key[2], "bit_depth": key[3], "channels": key[4]}
                for key, count in counter.most_common()
            ]
            for modality, counter in signatures.items()
        },
    }
    return records, summary


def grouped_multilabel_split(records: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for record in records:
        grouped[record["group"]].append(record)

    total_classes = [0] * NUM_CLASSES
    for record in records:
        for label in record["labels"]:
            total_classes[int(label[0])] += 1
    target_classes = [count * val_fraction for count in total_classes]
    target_samples = len(records) * val_fraction

    rng = random.Random(seed)
    groups = []
    for group_id, items in grouped.items():
        counts = [0] * NUM_CLASSES
        for item in items:
            for label in item["labels"]:
                counts[int(label[0])] += 1
        rarity = sum(counts[index] / max(total_classes[index], 1) for index in range(NUM_CLASSES))
        groups.append((group_id, items, counts, rarity, rng.random()))
    groups.sort(key=lambda item: (-item[3], -len(item[1]), item[4]))

    val_groups: set[str] = set()
    val_counts = [0] * NUM_CLASSES
    val_samples = 0

    def objective(samples: int, counts: list[int]) -> float:
        size_error = abs(samples - target_samples) / max(target_samples, 1)
        class_error = sum(
            abs(counts[index] - target_classes[index]) / max(target_classes[index], 1)
            for index in range(NUM_CLASSES)
        ) / NUM_CLASSES
        overshoot = max(0, samples - target_samples * 1.08) / max(target_samples, 1)
        return 0.45 * size_error + 0.55 * class_error + 2.0 * overshoot

    for group_id, items, counts, _, _ in groups:
        candidate_counts = [left + right for left, right in zip(val_counts, counts)]
        if objective(val_samples + len(items), candidate_counts) < objective(val_samples, val_counts):
            val_groups.add(group_id)
            val_samples += len(items)
            val_counts = candidate_counts

    # Guarantee every class with enough samples appears in validation.
    for class_id, total in enumerate(total_classes):
        if total == 0 or val_counts[class_id] > 0:
            continue
        candidates = [item for item in groups if item[0] not in val_groups and item[2][class_id] > 0]
        if candidates:
            group_id, items, counts, _, _ = min(candidates, key=lambda item: len(item[1]))
            val_groups.add(group_id)
            val_samples += len(items)
            val_counts = [left + right for left, right in zip(val_counts, counts)]

    train_records = [record for record in records if record["group"] not in val_groups]
    val_records = [record for record in records if record["group"] in val_groups]
    if not train_records or not val_records:
        raise RuntimeError("Grouped split produced an empty partition")
    return train_records, val_records


SCENE_CLASS_GROUPS = (
    ("uav", {10}),
    ("water", {1}),
    ("traffic", {5, 6, 11}),
    ("animal", {2}),
    ("recreation", {3, 7}),
    ("street_furniture", {4, 8, 9}),
)


def resolution_bucket(record: dict) -> str:
    """Return a stable resolution stratum from visible-image metadata."""
    metadata = record["metadata"]["visible"]
    width, height = int(metadata["width"]), int(metadata["height"])
    return f"{height}p_{width}x{height}"


def semantic_scene_bucket(record: dict) -> str:
    """Infer a coarse scene proxy from labels when explicit scene IDs are absent.

    Sequence groups remain atomic during splitting. This bucket only constrains the
    high-level semantic mix inside train/validation; it is not treated as a true
    human-annotated scene label.
    """
    classes = {int(label[0]) for label in record.get("labels", [])}
    for name, members in SCENE_CLASS_GROUPS:
        if classes & members:
            return name
    return "person_or_background"


def resolution_scene_grouped_split(
    records: list[dict], val_fraction: float, seed: int
) -> tuple[list[dict], list[dict], dict]:
    """Leakage-aware split balanced by resolution, scene proxy and image classes.

    Adjacent frames represented by ``record['group']`` are never separated. The
    greedy objective explicitly matches the rare 360p/UAV stratum instead of
    allowing a random split to use resolution as a shortcut for class 10.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if not records:
        raise ValueError("records must not be empty")

    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for record in records:
        grouped[record["group"]].append(record)

    def summarize(items: list[dict]) -> dict:
        resolutions = collections.Counter()
        scenes = collections.Counter()
        image_classes = collections.Counter()
        box_classes = collections.Counter()
        for item in items:
            resolution = resolution_bucket(item)
            scene = semantic_scene_bucket(item)
            resolutions[resolution] += 1
            scenes[f"{resolution}|{scene}"] += 1
            present = {int(label[0]) for label in item.get("labels", [])}
            image_classes.update(present)
            box_classes.update(int(label[0]) for label in item.get("labels", []))
        return {
            "samples": len(items),
            "resolutions": resolutions,
            "resolution_scenes": scenes,
            "image_classes": image_classes,
            "box_classes": box_classes,
        }

    total = summarize(records)
    targets = {
        name: {key: value * val_fraction for key, value in total[name].items()}
        for name in ("resolutions", "resolution_scenes", "image_classes", "box_classes")
    }
    target_samples = len(records) * val_fraction

    group_rows = []
    rng = random.Random(seed)
    for group_id, items in grouped.items():
        stats = summarize(items)
        rarity = sum(
            count / max(total["resolution_scenes"][key], 1)
            for key, count in stats["resolution_scenes"].items()
        ) + sum(
            count / max(total["image_classes"][key], 1)
            for key, count in stats["image_classes"].items()
        )
        group_rows.append((group_id, items, stats, rarity, rng.random()))
    group_rows.sort(key=lambda row: (-row[3], -len(row[1]), row[4]))

    current = {
        "samples": 0,
        "resolutions": collections.Counter(),
        "resolution_scenes": collections.Counter(),
        "image_classes": collections.Counter(),
        "box_classes": collections.Counter(),
    }

    def add_stats(left: dict, right: dict) -> dict:
        return {
            "samples": left["samples"] + right["samples"],
            **{
                name: left[name] + right[name]
                for name in ("resolutions", "resolution_scenes", "image_classes", "box_classes")
            },
        }

    def relative_error(values, expected) -> float:
        if not expected:
            return 0.0
        return sum(
            abs(values.get(key, 0) - target) / max(target, 1.0)
            for key, target in expected.items()
        ) / len(expected)

    def objective(stats: dict) -> float:
        size_error = abs(stats["samples"] - target_samples) / max(target_samples, 1.0)
        overshoot = max(0.0, stats["samples"] - target_samples * 1.02) / max(
            target_samples, 1.0
        )
        return (
            2.0 * size_error
            + 1.5 * relative_error(stats["resolutions"], targets["resolutions"])
            + 1.5 * relative_error(
                stats["resolution_scenes"], targets["resolution_scenes"]
            )
            + 1.0 * relative_error(stats["image_classes"], targets["image_classes"])
            + 0.25 * relative_error(stats["box_classes"], targets["box_classes"])
            + 5.0 * overshoot
        )

    selected: set[str] = set()
    remaining = list(group_rows)
    while remaining and current["samples"] < target_samples:
        best_index = min(
            range(len(remaining)),
            key=lambda index: (
                objective(add_stats(current, remaining[index][2])),
                -remaining[index][3],
                remaining[index][4],
            ),
        )
        group_id, _, stats, _, _ = remaining.pop(best_index)
        selected.add(group_id)
        current = add_stats(current, stats)

    train_records = [record for record in records if record["group"] not in selected]
    val_records = [record for record in records if record["group"] in selected]
    if not train_records or not val_records:
        raise RuntimeError("Resolution-scene grouped split produced an empty partition")

    train_groups = {record["group"] for record in train_records}
    val_groups = {record["group"] for record in val_records}
    overlap = sorted(train_groups & val_groups)
    if overlap:
        raise RuntimeError(f"Group leakage detected: {overlap[:5]}")

    def serializable(stats: dict) -> dict:
        return {
            "samples": stats["samples"],
            **{
                name: {str(key): int(value) for key, value in sorted(stats[name].items())}
                for name in ("resolutions", "resolution_scenes", "image_classes", "box_classes")
            },
        }

    audit = {
        "strategy": "sequence_grouped_resolution_x_semantic_scene_x_multilabel",
        "seed": seed,
        "val_fraction": val_fraction,
        "scene_note": "semantic scene is a label-derived proxy; sequence groups are atomic",
        "all": serializable(total),
        "train": serializable(summarize(train_records)),
        "val": serializable(summarize(val_records)),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "group_overlap": overlap,
    }
    return train_records, val_records, audit


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_manifests(data_root: Path, output_dir: Path, val_fraction: float, seed: int) -> dict:
    train_archives = sorted((data_root / "train").glob("*.zip"))
    test_archives = sorted((data_root / "test").glob("*.zip"))
    if len(train_archives) != 1 or len(test_archives) != 1:
        raise FileNotFoundError(
            f"Expected exactly one train and one test zip under {data_root}; "
            f"found train={train_archives}, test={test_archives}"
        )
    train_records, train_summary = scan_archive(train_archives[0], data_root, require_labels=True)
    test_records, test_summary = scan_archive(test_archives[0], data_root, require_labels=False)
    training, validation = grouped_multilabel_split(train_records, val_fraction, seed)

    write_jsonl(train_records, output_dir / "all_train.jsonl")
    write_jsonl(training, output_dir / "train.jsonl")
    write_jsonl(validation, output_dir / "val.jsonl")
    write_jsonl(test_records, output_dir / "test.jsonl")

    result = {
        "seed": seed,
        "val_fraction": val_fraction,
        "train_samples": len(training),
        "val_samples": len(validation),
        "train_groups": len({item["group"] for item in training}),
        "val_groups": len({item["group"] for item in validation}),
        "group_overlap": sorted({item["group"] for item in training} & {item["group"] for item in validation}),
        "train_archive": train_summary,
        "test_archive": test_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result
