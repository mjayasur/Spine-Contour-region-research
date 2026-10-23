"""Routing and numerical contracts; these are not radiograph validation."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from backend.decoders import local_coordinates
from backend.fixed_pipeline import crop_windows, cervical_geometry
from backend.measurement_routes import source_geometry, measure
from backend.pipeline_stages import production_stages
from backend.tests.unit.test_global_sva_measurements import geometry


def test_cervical_boundary_neighbors_are_unique():
    heat = np.zeros((1, 1, 9, 9), np.float32)
    # A flat local window at the top-left peak has coordinates 0 through 4.
    np.testing.assert_allclose(local_coordinates(heat, unique_boundary=True), [[[2., 2.]]])
    assert local_coordinates(heat)[0, 0, 0] < 2


def test_forced_extension_applies_to_every_candidate(monkeypatch):
    original = [(10, 100, 90, 200), (12, 20, 92, 180)]
    monkeypatch.setattr('backend.fixed_pipeline.full_spine.lumbar_windows', lambda *a: original)
    assert crop_windows(None, (300, 100), 'all_up15') == [(10, 85, 90, 200), (12, 0, 92, 180)]
    assert set(crop_windows(None, (300, 100), 'original_plus_up15')) == set(original + [(10, 85, 90, 200), (12, 0, 92, 180)])
    with pytest.raises(ValueError):
        crop_windows(None, (300, 100), 'best_reader_result')


def test_replacement_graph_keeps_explicit_decoder(monkeypatch):
    from backend.models import models
    calls = []
    monkeypatch.setattr(models, '_infer', lambda kind, operation, message: calls.append(kind) or np.zeros((1, 23, 96, 96)))
    # Invalid/out-of-crop points are deliberately withheld by the contract.
    decoder = lambda heat: np.full((23, 2), 40.)
    stages = replace(production_stages(), cervical_graph='replacement', cervical_decoder=decoder)
    result = cervical_geometry(np.zeros((100, 100), np.uint8), [0, 0, 100, 100], 'left',
                               stages.cervical_graph, stages.cervical_decoder)
    assert calls == ['replacement']
    assert result['c2_centroid'] is not None


def routed_geometry():
    g = geometry()
    g['source_geometry'] = deepcopy(g)
    g['measurement_sources'] = {'global_sva': {'c7_centroid': [60., 50.]},
                                'pelvic_tilt': {'s1_superior': [[20., 80.], [40., 90.]],
                                                'femoral_circles': [[30., 110., 5.], [50., 110., 5.]]}}
    return g


def test_route_anchors_do_not_mutate_common_geometry():
    g = routed_geometry()
    baseline = deepcopy(g)
    assert source_geometry(g, 'global_sva')['c7_centroid'] == [60., 50.]
    assert source_geometry(g, 'pelvic_tilt')['s1_superior'] == [[20., 80.], [40., 90.]]
    assert g == baseline


def test_manual_edits_and_clear_supersede_automatic_sources():
    g = routed_geometry()
    g['c7_centroid'] = None
    assert source_geometry(g, 'global_sva')['c7_centroid'] is None
    g['s1_superior'] = [[22., 81.], [40., 90.]]
    assert source_geometry(g, 'pelvic_tilt')['s1_superior'] == g['s1_superior']


def test_saved_route_sources_survive_measurement_roundtrip():
    g = routed_geometry()
    result = measure(g)
    assert result['geometry']['measurement_sources'] == g['measurement_sources']
    assert measure(result['geometry'])['measurements'] == result['measurements']
