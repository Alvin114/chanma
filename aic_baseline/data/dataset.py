from __future__ import annotations

import collections
import random
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from aic_baseline.constants import NUM_CLASSES
from aic_baseline.data.manifest import read_jsonl
from aic_baseline.data.transforms import (
    horizontal_flip,
    letterbox,
    photometric_augment,
    random_affine,
    xywhn_to_xyxy,
    xyxy_to_xywhn,
)


class AICDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path,
        image_size: int,
        input_mode: str = "rgbtd",
        augment: bool = False,
        augmentation: dict[str, Any] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest)
        self.records = read_jsonl(self.manifest_path)
        self.data_root = Path(data_root).resolve()
        self.image_size = int(image_size)
        self.input_mode = input_mode.lower()
        if self.input_mode not in {"rgb", "ir", "rgbt", "rgbd", "rgbtd"}:
            raise ValueError(f"Unsupported input_mode: {input_mode}")
        self.augment = augment
        self.augmentation = augmentation or {}
        self._zip_handles: dict[str, zipfile.ZipFile] = {}
        self.class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        self.image_classes: list[set[int]] = []
        for record in self.records:
            classes = {int(label[0]) for label in record.get("labels", [])}
            self.image_classes.append(classes)
            for label in record.get("labels", []):
                self.class_counts[int(label[0])] += 1

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zip_handles"] = {}
        return state

    def close(self) -> None:
        for handle in self._zip_handles.values():
            handle.close()
        self._zip_handles.clear()

    def _zip(self, relative_path: str) -> zipfile.ZipFile:
        if relative_path not in self._zip_handles:
            archive = self.data_root / relative_path
            self._zip_handles[relative_path] = zipfile.ZipFile(archive)
        return self._zip_handles[relative_path]

    def _decode(self, record: dict, modality: str) -> np.ndarray:
        raw = self._zip(record["archive"]).read(record["members"][modality])
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"OpenCV failed to decode {record['members'][modality]}")
        return image

    @staticmethod
    def _visible(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    @staticmethod
    def _infrared(image: np.ndarray) -> np.ndarray:
        gray = image if image.ndim == 2 else image[..., 0]
        gray = gray.astype(np.float32) / 255.0
        valid = (gray > 0).astype(np.float32)
        return np.stack((gray, valid), axis=2)

    @staticmethod
    def _depth(image: np.ndarray, metadata: dict) -> np.ndarray:
        is_metric = metadata["format"] == "png" and metadata["bit_depth"] == 16
        if is_metric:
            raw = image if image.ndim == 2 else image[..., 0]
            raw = raw.astype(np.float32)
            valid = ((raw >= 300) & (raw <= 20000)).astype(np.float32)
            clipped = np.clip(raw, 300, 20000)
            normalized = np.log(clipped / 300.0) / np.log(20000.0 / 300.0)
            normalized *= valid
            metric = np.ones_like(normalized, dtype=np.float32)
        else:
            gray = image if image.ndim == 2 else image[..., 0]
            normalized = gray.astype(np.float32) / 255.0
            valid = (gray > 0).astype(np.float32)
            metric = np.zeros_like(normalized, dtype=np.float32)
        return np.stack((normalized, valid, metric), axis=2)

    def __getitem__(self, index: int):
        record = self.records[index]
        images: dict[str, np.ndarray] = {}
        if self.input_mode != "ir":
            images["rgb"] = self._visible(self._decode(record, "visible"))
        if self.input_mode == "ir":
            images["infrared"] = self._infrared(self._decode(record, "infrared"))
        if "t" in self.input_mode:
            images["infrared"] = self._infrared(self._decode(record, "infrared"))
        if "d" in self.input_mode:
            images["depth"] = self._depth(self._decode(record, "depth"), record["metadata"]["depth"])

        original_height, original_width = next(iter(images.values())).shape[:2]
        raw_labels = np.asarray(record.get("labels", []), dtype=np.float32).reshape(-1, 5)
        classes, boxes = xywhn_to_xyxy(raw_labels, original_width, original_height)

        if self.augment:
            if random.random() < self.augmentation.get("affine_prob", 0.8):
                images, classes, boxes = random_affine(
                    images,
                    classes,
                    boxes,
                    degrees=float(self.augmentation.get("degrees", 3.0)),
                    translate=float(self.augmentation.get("translate", 0.08)),
                    scale=float(self.augmentation.get("scale", 0.25)),
                    shear=float(self.augmentation.get("shear", 1.0)),
                )
            if random.random() < self.augmentation.get("horizontal_flip", 0.5):
                images, boxes = horizontal_flip(images, boxes)
            images = photometric_augment(images, self.augmentation)
            if "infrared" in images and random.random() < self.augmentation.get("infrared_dropout", 0.0):
                images["infrared"] *= 0
            if "depth" in images and random.random() < self.augmentation.get("depth_dropout", 0.0):
                images["depth"][..., :2] *= 0

        images, boxes, ratio, pad = letterbox(images, boxes, self.image_size)
        labels = xyxy_to_xywhn(classes, boxes, self.image_size, self.image_size)
        tensors = {
            name: torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
            for name, image in images.items()
        }
        target = torch.from_numpy(labels).float()
        meta = {
            "id": record["id"],
            "original_shape": (original_height, original_width),
            "input_shape": (self.image_size, self.image_size),
            "ratio": ratio,
            "pad": pad,
        }
        return tensors, target, meta

    def sampling_weights(self, power: float = 0.5) -> torch.Tensor:
        frequencies = np.maximum(self.class_counts, 1)
        class_weights = (frequencies.sum() / frequencies) ** power
        class_weights /= class_weights.mean()
        weights = []
        for classes in self.image_classes:
            weights.append(max((class_weights[class_id] for class_id in classes), default=1.0))
        return torch.tensor(weights, dtype=torch.double)


def collate_batch(batch):
    image_dicts, targets, metas = zip(*batch)
    keys = image_dicts[0].keys()
    images = {key: torch.stack([item[key] for item in image_dicts], 0) for key in keys}
    batched_targets = []
    for batch_index, target in enumerate(targets):
        if len(target):
            indices = torch.full((len(target), 1), batch_index, dtype=target.dtype)
            batched_targets.append(torch.cat((indices, target), dim=1))
    merged = torch.cat(batched_targets, 0) if batched_targets else torch.zeros((0, 6), dtype=torch.float32)
    return images, merged, list(metas)

