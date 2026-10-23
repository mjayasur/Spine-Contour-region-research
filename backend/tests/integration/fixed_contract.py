"""Mocked API contracts only; no claim of model accuracy on generated pixels."""
from copy import deepcopy
import numpy as np
from backend import fixed_pipeline, runtime
from backend.tests.unit.test_partial_measurements import anatomy


def stub(monkeypatch, levels=('L1',), s1=None):
    vertebrae, _, _ = anatomy(levels)
    def prediction(pixels, anterior_side=None, *, region='lumbar', detection_evidence=None):
        runtime.report('landmarks', 'Finding lumbar landmarks')
        g = dict(region=region, anterior_side='left', vertebrae=deepcopy(vertebrae),
                 s1_superior=deepcopy(s1), femoral_circles=[], c2_centroid=None,
                 c7_centroid=None, pipeline_revision=fixed_pipeline.REVISION)
        return dict(image=pixels, mask=np.zeros_like(pixels), femoral_mask=np.zeros_like(pixels),
                    landmarks=g, models={'vertebrae': fixed_pipeline.REVISION}, framing={'searched': True})
    monkeypatch.setattr(fixed_pipeline, 'predict', prediction)
