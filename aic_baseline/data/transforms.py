from __future__ import annotations

import math
import random
from typing import Any

import cv2
import numpy as np


def xywhn_to_xyxy(labels: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    if labels.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    classes = labels[:, 0].astype(np.int64)
    boxes = np.empty((len(labels), 4), dtype=np.float32)
    boxes[:, 0] = (labels[:, 1] - labels[:, 3] / 2) * width
    boxes[:, 1] = (labels[:, 2] - labels[:, 4] / 2) * height
    boxes[:, 2] = (labels[:, 1] + labels[:, 3] / 2) * width
    boxes[:, 3] = (labels[:, 2] + labels[:, 4] / 2) * height
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return classes, boxes


def xyxy_to_xywhn(classes: np.ndarray, boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    result = np.empty((len(boxes), 5), dtype=np.float32)
    result[:, 0] = classes
    result[:, 1] = (boxes[:, 0] + boxes[:, 2]) / 2 / width
    result[:, 2] = (boxes[:, 1] + boxes[:, 3]) / 2 / height
    result[:, 3] = (boxes[:, 2] - boxes[:, 0]) / width
    result[:, 4] = (boxes[:, 3] - boxes[:, 1]) / height
    valid = (result[:, 3] > 1e-5) & (result[:, 4] > 1e-5)
    return result[valid]


def random_affine(
    images: dict[str, np.ndarray],
    classes: np.ndarray,
    boxes: np.ndarray,
    degrees: float,
    translate: float,
    scale: float,
    shear: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    height, width = next(iter(images.values())).shape[:2]
    angle = random.uniform(-degrees, degrees)
    scale_factor = random.uniform(1 - scale, 1 + scale)
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale_factor)
    matrix = np.eye(3, dtype=np.float32)
    matrix[:2] = rotation

    shear_matrix = np.eye(3, dtype=np.float32)
    shear_matrix[0, 1] = math.tan(math.radians(random.uniform(-shear, shear)))
    shear_matrix[1, 0] = math.tan(math.radians(random.uniform(-shear, shear)))
    translate_matrix = np.eye(3, dtype=np.float32)
    translate_matrix[0, 2] = random.uniform(-translate, translate) * width
    translate_matrix[1, 2] = random.uniform(-translate, translate) * height
    matrix = translate_matrix @ shear_matrix @ matrix

    warped = {}
    for name, image in images.items():
        fill: float | tuple[float, ...]
        if name == "rgb":
            fill = tuple([114 / 255.0] * image.shape[2])
        else:
            fill = tuple([0.0] * image.shape[2])
        warped[name] = cv2.warpPerspective(
            image,
            matrix,
            dsize=(width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill,
        )
        if warped[name].ndim == 2:
            warped[name] = warped[name][..., None]

    if boxes.size == 0:
        return warped, classes, boxes
    corners = np.ones((len(boxes) * 4, 3), dtype=np.float32)
    corners[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(-1, 2)
    transformed = (corners @ matrix.T)
    transformed = transformed[:, :2] / transformed[:, 2:3].clip(1e-6)
    transformed = transformed.reshape(len(boxes), 8)
    xs, ys = transformed[:, 0::2], transformed[:, 1::2]
    new_boxes = np.stack((xs.min(1), ys.min(1), xs.max(1), ys.max(1)), axis=1)
    new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clip(0, width)
    new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clip(0, height)

    old_wh = boxes[:, 2:4] - boxes[:, 0:2]
    new_wh = new_boxes[:, 2:4] - new_boxes[:, 0:2]
    area_ratio = (new_wh[:, 0] * new_wh[:, 1]) / (old_wh[:, 0] * old_wh[:, 1] * scale_factor**2 + 1e-6)
    aspect = np.maximum(new_wh[:, 0] / (new_wh[:, 1] + 1e-6), new_wh[:, 1] / (new_wh[:, 0] + 1e-6))
    keep = (new_wh[:, 0] > 2) & (new_wh[:, 1] > 2) & (area_ratio > 0.1) & (aspect < 20)
    return warped, classes[keep], new_boxes[keep]


def horizontal_flip(
    images: dict[str, np.ndarray], boxes: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    width = next(iter(images.values())).shape[1]
    flipped = {name: np.ascontiguousarray(image[:, ::-1]) for name, image in images.items()}
    if boxes.size:
        boxes = boxes.copy()
        left = width - boxes[:, 2]
        right = width - boxes[:, 0]
        boxes[:, 0], boxes[:, 2] = left, right
    return flipped, boxes


def photometric_augment(images: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, np.ndarray]:
    images = {name: image.copy() for name, image in images.items()}
    if "rgb" in images:
        brightness = random.uniform(1 - config.get("rgb_brightness", 0.15), 1 + config.get("rgb_brightness", 0.15))
        contrast = random.uniform(1 - config.get("rgb_contrast", 0.15), 1 + config.get("rgb_contrast", 0.15))
        mean = images["rgb"].mean(axis=(0, 1), keepdims=True)
        images["rgb"] = np.clip((images["rgb"] - mean) * contrast + mean, 0, 1) * brightness
        images["rgb"] = np.clip(images["rgb"], 0, 1)
    if "infrared" in images:
        gain = random.uniform(0.85, 1.15)
        bias = random.uniform(-0.05, 0.05)
        images["infrared"][..., 0] = np.clip(images["infrared"][..., 0] * gain + bias, 0, 1)
    if "depth" in images and random.random() < config.get("depth_hole_prob", 0.25):
        height, width = images["depth"].shape[:2]
        hole_w = random.randint(max(1, width // 30), max(2, width // 8))
        hole_h = random.randint(max(1, height // 30), max(2, height // 8))
        x1 = random.randint(0, max(0, width - hole_w))
        y1 = random.randint(0, max(0, height - hole_h))
        images["depth"][y1:y1 + hole_h, x1:x1 + hole_w, :2] = 0
    return images


def letterbox(
    images: dict[str, np.ndarray], boxes: np.ndarray, image_size: int
) -> tuple[dict[str, np.ndarray], np.ndarray, float, tuple[float, float]]:
    height, width = next(iter(images.values())).shape[:2]
    ratio = min(image_size / height, image_size / width)
    resized_width, resized_height = int(round(width * ratio)), int(round(height * ratio))
    pad_w = (image_size - resized_width) / 2
    pad_h = (image_size - resized_height) / 2
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))

    output = {}
    for name, image in images.items():
        interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
        if resized.ndim == 2:
            resized = resized[..., None]
        fill = tuple([114 / 255.0] * resized.shape[2]) if name == "rgb" else tuple([0.0] * resized.shape[2])
        output[name] = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill
        )
        if output[name].ndim == 2:
            output[name] = output[name][..., None]
    if boxes.size:
        boxes = boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio + left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio + top
    return output, boxes, ratio, (float(left), float(top))

