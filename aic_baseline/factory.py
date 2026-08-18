from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from aic_baseline.data import AICDataset, collate_batch
from aic_baseline.models import MultiModalYOLO
from aic_baseline.utils import worker_seed


def build_model(config: dict) -> MultiModalYOLO:
    model_config = config["model"]
    return MultiModalYOLO(
        input_mode=model_config.get("input_mode", "rgbtd"),
        num_classes=int(model_config.get("num_classes", 12)),
        width=float(model_config.get("width", 0.5)),
        depth=float(model_config.get("depth", 0.33)),
        fusion_heads=int(model_config.get("fusion_heads", 8)),
        fusion_loops=int(model_config.get("fusion_loops", 1)),
        fusion_dropout=float(model_config.get("fusion_dropout", 0.1)),
        fusion_type=str(model_config.get("fusion_type", "ica")),
    )


def build_dataset(config: dict, split: str, augment: bool) -> AICDataset:
    data_config = config["data"]
    return AICDataset(
        manifest=data_config[f"{split}_manifest"],
        data_root=data_config.get("root", "data"),
        image_size=int(data_config.get("image_size", 960)),
        input_mode=config["model"].get("input_mode", "rgbtd"),
        augment=augment,
        augmentation=config.get("augmentation", {}),
    )


def build_dataloader(config: dict, split: str, augment: bool, seed: int | None = None):
    dataset = build_dataset(config, split, augment)
    train_config = config.get("train", {})
    is_train = split == "train"
    batch_size = int(train_config.get("batch_size", 8) if is_train else train_config.get("val_batch_size", train_config.get("batch_size", 8)))
    workers = int(data_config_value(config, "workers", 4))
    sampler = None
    shuffle = is_train
    sampling_power = float(train_config.get("class_aware_sampling_power", 0.0))
    generator = torch.Generator()
    generator.manual_seed(int(seed if seed is not None else train_config.get("seed", 3407)))
    if is_train and sampling_power > 0:
        weights = dataset.sampling_weights(sampling_power)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
        shuffle = False
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": is_train,
        "collate_fn": collate_batch,
        "worker_init_fn": worker_seed,
        "generator": generator,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(data_config_value(config, "prefetch_factor", 2))
    return DataLoader(**kwargs), dataset


def data_config_value(config: dict, key: str, default):
    return config.get("data", {}).get(key, default)

