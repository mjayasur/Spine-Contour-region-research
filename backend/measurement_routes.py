"""Measurement-specific source geometry, including interactive corrections."""
from copy import deepcopy
from .global_sva_measurements import global_sva_measurements_from_geometry


def source_geometry(geometry, route):
    result = deepcopy(geometry)
    sources = result.pop('measurement_sources', {})
    baseline = result.pop('source_geometry', None)
    source = sources.get(route)
    if source is None:
        return result
    # A human correction supersedes automatic route-specific anchors. Never
    # restore old automatic landmarks over an edited/cleared source field.
    fields = ('s1_superior', 'c7_centroid') if route == 'global_sva' else ('s1_superior', 'femoral_circles')
    def field(record, key):
        value = record.get(key)
        return (value or []) if key == 'femoral_circles' else value
    if baseline is not None and any(field(result, k) != field(baseline, k) for k in fields):
        return result
    for key, value in source.items():
        if key != 'vertebrae':
            result[key] = deepcopy(value)
    return result


def measure(geometry):
    from .cervical_measurements import cervical_measurements_from_geometry
    if geometry.get('region') == 'cervical':
        return cervical_measurements_from_geometry(geometry)
    region = geometry.get('region', 'full_spine')
    geometry = {**geometry, 'region': 'full_spine'}
    result = global_sva_measurements_from_geometry(geometry)
    if geometry.get('measurement_sources'):
        global_result = global_sva_measurements_from_geometry(source_geometry(geometry, 'global_sva'))
        pelvic_result = global_sva_measurements_from_geometry(source_geometry(geometry, 'pelvic_tilt'))
        for key in ('GLOBAL_SVA_PX', 'GLOBAL_SVA_MM'):
            result['measurements'][key] = global_result['measurements'][key]
        result['measurements']['PT'] = pelvic_result['measurements']['PT']
        result['qc']['global_sva'] = global_result['qc']['global_sva']
    result['geometry']['region'] = region
    result['measurements']['region'] = region
    return result
