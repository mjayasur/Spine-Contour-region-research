// The selected measurement's automatic anchors can differ from the common
// editable geometry. Manual corrections supersede the affected automatic route.
export function measurementGeometry(geometry, route) {
  if (!geometry) return geometry;
  const source = geometry.measurement_sources?.[route];
  if (!source) return geometry;
  const fields = route === 'global_sva' ? ['s1_superior', 'c7_centroid'] : ['s1_superior', 'femoral_circles'];
  const baseline = geometry.source_geometry;
  const field = (record, key) => record[key] ?? (key === 'femoral_circles' ? [] : null);
  if (baseline && fields.some(key => JSON.stringify(field(geometry, key)) !== JSON.stringify(field(baseline, key)))) return geometry;
  const { vertebrae, ...anchors } = source;
  const result = { ...geometry, ...anchors };
  if (route === 'pelvic_tilt') {
    const circles = result.femoral_circles ?? [];
    result.hip_midpoint = circles.length === 2 ? [0, 1].map(i => (circles[0][i]+circles[1][i])/2) : null;
  }
  return result;
}
