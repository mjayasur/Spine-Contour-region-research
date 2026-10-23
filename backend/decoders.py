"""The two frozen lumbar decoders and the cervical subpixel decoder."""
import numpy as np


def local_coordinates(heat, *, radius=4, temperature=12., positive_weights=False, unique_boundary=False):
    heat = np.asarray(heat, dtype=np.float32)
    if heat.ndim != 4 or not np.isfinite(heat).all():
        raise ValueError("Expected finite NCHW landmark heatmaps")
    n, k, h, w = heat.shape
    flat = heat.reshape(n, k, -1)
    peak = flat.argmax(-1)
    dy, dx = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1), indexing='ij')
    raw_x, raw_y = peak[..., None] % w + dx.ravel(), peak[..., None] // w + dy.ravel()
    x, y = np.clip(raw_x, 0, w-1), np.clip(raw_y, 0, h-1)
    weights = np.take_along_axis(flat, y*w+x, axis=-1)
    if positive_weights:
        weights = np.maximum(weights, 0)
    else:
        weights = weights * np.float32(temperature)
        weights = np.exp(weights-weights.max(-1, keepdims=True))
    if unique_boundary:
        weights *= (raw_x >= 0) & (raw_x < w) & (raw_y >= 0) & (raw_y < h)
    weights /= np.maximum(weights.sum(-1, keepdims=True), np.float32(1e-6))
    return np.stack(((weights*x.astype(np.float32)).sum(-1),
                     (weights*y.astype(np.float32)).sum(-1)), -1)


def lumbar_points(heat, decoder='subpixel'):
    if decoder == 'original':
        return local_coordinates(heat, radius=3, positive_weights=True)*4
    if decoder != 'subpixel':
        raise ValueError('Unknown lumbar decoder')
    return local_coordinates(heat)*4
