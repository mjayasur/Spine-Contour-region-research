"""Replaceable stage interfaces for the fixed production measurement recipe.

Detectors return canonical-image crop boxes, predictors return model-frame
arrays, and all public geometry is restored to original-image coordinates.
Model replacement changes this configuration, not the measurement routing.
"""
from dataclasses import dataclass
from typing import Callable
from . import framing
from .models import models, full_spine
from .decoders import local_coordinates, lumbar_points


def cervical_subpixel(heatmaps):
    return local_coordinates(heatmaps, unique_boundary=True)[0]


def cervical_original(heatmaps):
    from .models.cervical_anatomical import decode_heatmaps
    return decode_heatmaps(heatmaps)[0]


def neck_crop(image):
    from .models import cervical
    image = image if image.dtype.name == 'uint8' else models._robust_rescale(image)
    logits, boxes = models._infer('cervical_detr', lambda session: session.run(None, cervical.detector_input(image)),
                                 'Locating the cervical crop')
    found = cervical.detection_from_output(logits, boxes, image.shape)
    prepared = cervical.landmark_input(image, found['bbox']) if found else None
    return list(prepared[1].window) if prepared else None


@dataclass(frozen=True)
class Stages:
    lumbar_locator: Callable
    cervical_candidates: Callable
    sacrum_detector: Callable
    femoral_segmenter: Callable
    cervical_locator: Callable = neck_crop
    cervical_graph: str = 'cervical_slope_hrnet'
    global_cervical_graph: str = 'cervical_anatomical_hrnet'
    lumbar_graph: str = 'lumbar_heatmaps'
    global_lumbar_graph: str = 'hrnet'
    cervical_decoder: Callable = cervical_subpixel
    global_cervical_decoder: Callable = cervical_original
    lumbar_decoder: Callable = lumbar_points


def production_stages():
    return Stages(lumbar_locator=lambda image: framing.locate(image, models._score_s1),
                  cervical_candidates=full_spine._cervical_candidates,
                  sacrum_detector=models._score_s1,
                  femoral_segmenter=full_spine._femoral_region)
