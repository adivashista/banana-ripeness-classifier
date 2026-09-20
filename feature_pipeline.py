"""Shared image preprocessing and feature extraction for banana classification."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

IMAGE_SIZE = (224, 224)
CLASSES = ("raw", "ripe", "overripe", "rotten")
FEATURE_VERSION = "v2_masked_color_texture_spatial"


def _largest_reasonable_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    image_area = mask.shape[0] * mask.shape[1]
    candidates = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= image_area * 0.003:
            candidates.append((area, label))
    if not candidates:
        return mask
    _, best_label = max(candidates)
    result = np.zeros_like(mask)
    result[labels == best_label] = 255
    return result


def create_banana_mask(image: np.ndarray) -> np.ndarray:
    """Create a robust foreground mask without using the class label."""
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_blur = cv2.GaussianBlur(hsv, (5, 5), 0)

    # Broad banana-colour candidates. These are only used to initialise the mask.
    green = cv2.inRange(hsv_blur, np.array([25, 20, 20]), np.array([95, 255, 255]))
    yellow = cv2.inRange(hsv_blur, np.array([10, 35, 35]), np.array([42, 255, 255]))
    brown = cv2.inRange(hsv_blur, np.array([0, 20, 10]), np.array([25, 255, 230]))
    candidate = cv2.bitwise_or(cv2.bitwise_or(green, yellow), brown)

    # Remove small noise and close gaps inside the banana silhouette.
    kernel = np.ones((7, 7), np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    candidate = _largest_reasonable_component(candidate)

    # GrabCut refines the rough colour mask. If it fails, use the rough mask.
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    mask[candidate == 255] = cv2.GC_PR_FGD
    border = max(2, min(h, w) // 40)
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD
    mask[candidate == 255] = cv2.GC_PR_FGD

    # Use a central seed when the colour mask is empty or fragmented.
    if cv2.countNonZero(candidate) < int(h * w * 0.01):
        mask[h // 5: 4 * h // 5, w // 5: 4 * w // 5] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
        refined = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        refined = _largest_reasonable_component(refined)
        if cv2.countNonZero(refined) >= int(h * w * 0.01):
            return refined
    except cv2.error:
        pass

    return candidate if cv2.countNonZero(candidate) else np.full((h, w), 255, np.uint8)


def _normalised_hist(channel: np.ndarray, mask: np.ndarray, bins: int, max_value: int) -> np.ndarray:
    hist = cv2.calcHist([channel], [0], mask, [bins], [0, max_value])
    hist = cv2.normalize(hist, hist).flatten() if hist is not None else np.zeros(bins, dtype=np.float32)
    return hist.astype(np.float32)


def extract_features(image: np.ndarray) -> np.ndarray:
    """Return the same feature vector for training, CLI inference, and FastAPI."""
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    mask = create_banana_mask(image)
    mask_bool = mask > 0
    pixel_count = max(int(mask_bool.sum()), 1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Masked mean/std prevent most background pixels from dominating the vector.
    pixels_bgr = image[mask_bool].astype(np.float32) if mask_bool.any() else image.reshape(-1, 3).astype(np.float32)
    pixels_hsv = hsv[mask_bool].astype(np.float32) if mask_bool.any() else hsv.reshape(-1, 3).astype(np.float32)
    pixels_lab = lab[mask_bool].astype(np.float32) if mask_bool.any() else lab.reshape(-1, 3).astype(np.float32)

    stats = np.concatenate([
        pixels_bgr.mean(axis=0), pixels_bgr.std(axis=0),
        pixels_hsv.mean(axis=0), pixels_hsv.std(axis=0),
        pixels_lab.mean(axis=0), pixels_lab.std(axis=0),
    ])

    histograms = []
    for channel, bins, max_value in [
        (image[:, :, 0], 16, 256), (image[:, :, 1], 16, 256), (image[:, :, 2], 16, 256),
        (hsv[:, :, 0], 18, 180), (hsv[:, :, 1], 16, 256), (hsv[:, :, 2], 16, 256),
        (lab[:, :, 0], 16, 256), (lab[:, :, 1], 16, 256), (lab[:, :, 2], 16, 256),
    ]:
        histograms.append(_normalised_hist(channel, mask, bins, max_value))

    # Texture and shape features, computed mostly inside the detected foreground.
    edges = cv2.Canny(gray, 60, 160)
    edge_density = np.array([float((edges[mask_bool] > 0).mean()) if mask_bool.any() else float((edges > 0).mean())])
    area_ratio = np.array([float(pixel_count / (IMAGE_SIZE[0] * IMAGE_SIZE[1]))])
    x, y, bw, bh = cv2.boundingRect(mask)
    aspect_ratio = np.array([float(bw / max(bh, 1))])

    # Coarse spatial colour means preserve where colour appears on the banana.
    spatial = []
    for gy in range(2):
        for gx in range(2):
            y0, y1 = gy * 112, (gy + 1) * 112
            x0, x1 = gx * 112, (gx + 1) * 112
            local_mask = mask[y0:y1, x0:x1] > 0
            local_pixels = image[y0:y1, x0:x1][local_mask]
            if len(local_pixels) == 0:
                local_pixels = np.zeros((1, 3), dtype=np.uint8)
            spatial.extend(local_pixels.mean(axis=0))

    vector = np.concatenate([stats, *histograms, edge_density, area_ratio, aspect_ratio, np.asarray(spatial)])
    return vector.astype(np.float32)


def extract_features_from_path(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return extract_features(image)
