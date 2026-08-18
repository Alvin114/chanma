from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from aic_baseline.config import save_config
from aic_baseline.engine.checkpoint import ModelEMA, load_checkpoint, save_checkpoint
from aic_baseline.engine.evaluate import evaluate
from aic_baseline.factory import build_dataloader, build_model
from aic_baseline.models import M2DDistillationLoss, YoloLoss, load_model_weights
from aic_baseline.utils import parameter_count, seed_everything, select_device


def _optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    train_config = config["train"]
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias") or ".bn." in name or "position_" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": float(train_config.get("weight_decay", 5e-4))},
    ]
    name = str(train_config.get("optimizer", "SGD")).lower()
    learning_rate = float(train_config.get("learning_rate", 0.01))
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(float(train_config.get("momentum", 0.9)), 0.999))
    if name == "sgd":
        return torch.optim.SGD(groups, lr=learning_rate, momentum=float(train_config.get("momentum", 0.937)), nesterov=True)
    raise ValueError(f"Unsupported optimizer: {name}")


def _scheduler(optimizer: torch.optim.Optimizer, config: dict):
    epochs = int(config["train"]["epochs"])
    warmup = int(config["train"].get("warmup_epochs", 3))
    final_factor = float(config["train"].get("final_lr_factor", 0.01))

    def schedule(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return max((epoch + 1) / warmup, 0.05)
        progress = (epoch - warmup) / max(epochs - warmup - 1, 1)
        progress = min(max(progress, 0.0), 1.0)
        return final_factor + (1 - final_factor) * (1 + math.cos(math.pi * progress)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)



def _load_teacher(path: str | Path, expected_mode: str, device: torch.device) -> nn.Module:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{expected_mode.upper()} teacher checkpoint not found: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, device)
    teacher_config = checkpoint.get("config")
    if not isinstance(teacher_config, dict):
        raise ValueError(f"Teacher checkpoint has no saved config: {checkpoint_path}")
    teacher = build_model(teacher_config).to(device)
    if teacher.input_mode != expected_mode:
        raise ValueError(
            f"Expected an {expected_mode!r} teacher, but {checkpoint_path} contains {teacher.input_mode!r}"
        )
    teacher.load_state_dict(checkpoint["model_state"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    print(f"loaded frozen {expected_mode.upper()} teacher from {checkpoint_path}")
    return teacher

def _checkpoint_state(model, ema, optimizer, scheduler, scaler, config, epoch, best_metric):
    return {
        "format_version": 1,
        "epoch": epoch,
        "best_map50_95": best_metric,
        "model_state": model.state_dict(),
        "ema_state": ema.ema.state_dict(),
        "ema_updates": ema.updates,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    }


def train(config: dict) -> Path:
    train_config = config["train"]
    seed = int(train_config.get("seed", 3407))
    seed_everything(seed, bool(train_config.get("deterministic", False)))
    device = select_device(str(train_config.get("device", "auto")))
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(train_config.get("cpu_threads", 8))))

    run_dir = Path(train_config.get("output_dir", "runs/baseline"))
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    model = build_model(config).to(device)
    total, trainable = parameter_count(model)
    print(f"device={device} parameters={total:,} trainable={trainable:,} input_mode={model.input_mode}")

    resume_path = train_config.get("resume")
    pretrained = train_config.get("pretrained")
    resume_checkpoint = None
    if resume_path:
        resume_checkpoint = load_checkpoint(resume_path, device)
        model.load_state_dict(resume_checkpoint["model_state"], strict=True)
        print(f"resumed model from {resume_path}")
    elif pretrained:
        pretrained_path = Path(pretrained)
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {pretrained_path}. "
                "Download it as documented in README.md or override train.pretrained=null."
            )
        information = load_model_weights(model, pretrained_path, device)
        print(f"loaded {information['loaded']} compatible tensors from {pretrained_path}")
    else:
        model.initialize_auxiliary_backbones()
        print("warning: training from random initialization")

    distillation_config = config.get("distillation", {})
    distillation_enabled = bool(distillation_config.get("enabled", False))
    teacher_rgb = teacher_ir = None
    distillation_loss = None
    if distillation_enabled:
        if model.input_mode == "ir" or "t" not in model.input_mode:
            raise ValueError("M2D dual-teacher distillation requires RGB and infrared student branches")
        teacher_rgb = _load_teacher(distillation_config["rgb_teacher"], "rgb", device)
        teacher_ir = _load_teacher(distillation_config["ir_teacher"], "ir", device)
        distillation_loss = M2DDistillationLoss(
            temperature=float(distillation_config.get("temperature", 1.0)),
            normal=bool(distillation_config.get("normal", True)),
            cross=bool(distillation_config.get("cross", True)),
        )

    train_loader, train_dataset = build_dataloader(config, "train", augment=True, seed=seed)
    val_loader, val_dataset = build_dataloader(config, "val", augment=False, seed=seed)
    loss_function = YoloLoss(model, config.get("loss", {}))
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config)
    amp_enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema = ModelEMA(model, float(train_config.get("ema_decay", 0.9999)))

    start_epoch = 0
    best_metric = -1.0
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        scaler.load_state_dict(resume_checkpoint.get("scaler_state", {}))
        if "ema_state" in resume_checkpoint:
            ema.ema.load_state_dict(resume_checkpoint["ema_state"])
            ema.updates = int(resume_checkpoint.get("ema_updates", 0))
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_metric = float(resume_checkpoint.get("best_map50_95", -1.0))

    epochs = int(train_config["epochs"])
    accumulation = max(1, int(train_config.get("gradient_accumulation", 1)))
    clip_norm = float(train_config.get("gradient_clip_norm", 10.0))
    val_interval = max(1, int(train_config.get("val_interval", 1)))
    patience = int(train_config.get("patience", 30))
    stale_epochs = 0
    log_file = run_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            means = torch.zeros(7, device=device)
            progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
            epoch_start = time.perf_counter()
            configured_max_batches = train_config.get("max_train_batches")
            epoch_batches = len(train_loader)
            if configured_max_batches is not None:
                epoch_batches = min(epoch_batches, max(0, int(configured_max_batches)))
            for step, (images, targets, _) in enumerate(progress):
                if step >= epoch_batches:
                    break
                images = {key: value.to(device, non_blocking=True) for key, value in images.items()}
                targets = targets.to(device, non_blocking=True)
                accumulation_window_start = (step // accumulation) * accumulation
                accumulation_window = min(accumulation, epoch_batches - accumulation_window_start)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    if distillation_enabled:
                        predictions, auxiliary = model(images, return_aux=True)
                    else:
                        predictions = model(images)
                        auxiliary = None
                    detection_loss, components = loss_function(predictions, targets)
                    intra_weighted = detection_loss.new_zeros(())
                    cross_weighted = detection_loss.new_zeros(())
                    illumination_weighted = detection_loss.new_zeros(())
                    if distillation_enabled:
                        with torch.no_grad():
                            teacher_rgb_features = teacher_rgb.extract_modal_features({"rgb": images["rgb"]})["rgb"]
                            teacher_ir_features = teacher_ir.extract_modal_features(
                                {"infrared": images["infrared"]}
                            )["ir"]
                        intra_loss, cross_loss = distillation_loss(
                            auxiliary["modal_features"]["rgb"],
                            auxiliary["modal_features"]["ir"],
                            teacher_rgb_features,
                            teacher_ir_features,
                        )
                        distill_decay = (
                            (1 - math.cos(epoch * math.pi / max(epochs, 1)))
                            / 2
                            * (float(distillation_config.get("final_factor", 0.1)) - 1)
                            + 1
                        )
                        distill_scale = float(distillation_config.get("weight", 0.8)) * distill_decay
                        intra_weighted = intra_loss * distill_scale
                        cross_weighted = cross_loss * distill_scale
                        illumination = auxiliary["illumination"]
                        if illumination is None:
                            raise RuntimeError("LIF illumination output is required when distillation is enabled")
                        illumination_target = F.adaptive_avg_pool2d(
                            images["rgb"].amax(dim=1, keepdim=True).detach(),
                            illumination.shape[-2:],
                        )
                        illumination_weighted = (
                            F.l1_loss(illumination.float(), illumination_target.float())
                            * float(distillation_config.get("illumination_weight", 1.3))
                        )
                    loss = detection_loss + intra_weighted + cross_weighted + illumination_weighted
                    # The last window can be shorter than `accumulation` (and the
                    # debug batch limit can make it very short). Normalizing by
                    # its actual size keeps that optimizer update correctly scaled.
                    scaled_loss = loss / accumulation_window
                scaler.scale(scaled_loss).backward()
                if (step + 1) % accumulation == 0 or step + 1 == epoch_batches:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)
                means[:3] = (means[:3] * step + components) / (step + 1)
                means[3] = (means[3] * step + loss.detach()) / (step + 1)
                means[4] = (means[4] * step + intra_weighted.detach()) / (step + 1)
                means[5] = (means[5] * step + cross_weighted.detach()) / (step + 1)
                means[6] = (means[6] * step + illumination_weighted.detach()) / (step + 1)
                progress.set_postfix(
                    box=f"{means[0]:.4f}",
                    obj=f"{means[1]:.4f}",
                    cls=f"{means[2]:.4f}",
                    distill=f"{means[4] + means[5]:.4f}",
                )
            scheduler.step()

            validation = None
            if (epoch + 1) % val_interval == 0 or epoch + 1 == epochs:
                validation = evaluate(
                    ema.ema,
                    val_loader,
                    device,
                    confidence_threshold=float(config.get("inference", {}).get("val_confidence", 0.001)),
                    iou_threshold=float(config.get("inference", {}).get("nms_iou", 0.65)),
                    max_detections=int(config.get("inference", {}).get("max_detections", 100)),
                    output_path=run_dir / "val_metrics_latest.json",
                    max_batches=(int(train_config["max_val_batches"]) if train_config.get("max_val_batches") is not None else None),
                )
                current = float(validation["map50_95"])
                if current > best_metric:
                    best_metric = current
                    stale_epochs = 0
                    best_state = _checkpoint_state(model, ema, optimizer, scheduler, scaler, config, epoch, best_metric)
                    best_state["model_state"] = ema.ema.state_dict()
                    save_checkpoint(best_state, run_dir / "best.pt")
                else:
                    stale_epochs += val_interval

            state = _checkpoint_state(model, ema, optimizer, scheduler, scaler, config, epoch, best_metric)
            save_checkpoint(state, run_dir / "last.pt")
            record = {
                "epoch": epoch + 1,
                "seconds": time.perf_counter() - epoch_start,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_box_loss": float(means[0]),
                "train_object_loss": float(means[1]),
                "train_class_loss": float(means[2]),
                "train_total_loss": float(means[3]),
                "train_intra_distillation_loss": float(means[4]),
                "train_cross_distillation_loss": float(means[5]),
                "train_illumination_loss": float(means[6]),
                "validation": validation,
            }
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary = f"epoch={epoch + 1} loss={means[3]:.4f}"
            if validation:
                summary += f" map50={validation['map50']:.4f} map50_95={validation['map50_95']:.4f} best={best_metric:.4f}"
            print(summary)
            if patience > 0 and stale_epochs >= patience:
                print(f"early stopping after {stale_epochs} validation epochs without improvement")
                break
    finally:
        train_dataset.close()
        val_dataset.close()
    return run_dir
