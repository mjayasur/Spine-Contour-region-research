import { measurementGeometry } from './measurement-sources.js';
import { boundCalibration, studyRegion, validAnteriorSide } from './cervical.js';

export const GLOBAL_SVA_COLUMNS = [
  { key: 'GLOBAL_SVA_MM', label: 'C7–S1 SVA (mm)', unit: 'mm' },
  { key: 'GLOBAL_SVA_PX', label: 'C7–S1 SVA (px)', unit: 'px' },
];

// Derive display/export from the current source-bound scale. A scale cleared by
// the user must never be replaced by an older value in the prediction geometry.
export function globalSvaMeasurements(study) {
  const values = { region: 'full_spine', GLOBAL_SVA_PX: null, GLOBAL_SVA_MM: null };
  if (studyRegion(study) !== 'full_spine' || !study?.geometry) return values;
  const g = measurementGeometry(study.geometry, 'global_sva');
  const point = p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite)
    && p[0] >= 0 && p[1] >= 0 && (!g.image_width || p[0] < g.image_width)
    && (!g.image_height || p[1] < g.image_height);
  const plate = g.s1_superior;
  if (!point(g.c7_centroid) || !Array.isArray(plate) || plate.length !== 2
      || !plate.every(point) || Math.hypot(plate[1][0] - plate[0][0], plate[1][1] - plate[0][1]) === 0
      || !validAnteriorSide(g.anterior_side)) return values;
  values.GLOBAL_SVA_PX = (g.c7_centroid[0] - plate[1][0]) * (g.anterior_side === 'left' ? -1 : 1) || 0;
  const calibration = boundCalibration(g, study.calibration);
  if (calibration?.spacing && [g.c7_centroid, ...plate].every(p => p[0] < calibration.width && p[1] < calibration.height)) {
    values.GLOBAL_SVA_MM = values.GLOBAL_SVA_PX * calibration.spacing.column_mm;
  }
  return values;
}

export function globalSvaRows(study, selectedLevel = null) {
  const m = globalSvaMeasurements(study);
  const mm = m.GLOBAL_SVA_MM !== null;
  const value = mm ? m.GLOBAL_SVA_MM : m.GLOBAL_SVA_PX;
  return [{ key: 'GLOBAL_SVA', label: mm ? 'C7–S1 SVA' : 'C7–S1 SVA · UNCALIBRATED',
    value, unit: mm ? 'mm' : 'px', absent: value === null, highlight: selectedLevel === 'GLOBAL_SVA' }];
}

export function globalSvaMeasureGeometry(geometry, calibration) {
  const scale = boundCalibration(geometry, calibration)?.spacing;
  return { ...geometry, pixel_spacing: scale ? [scale.row_mm, scale.column_mm] : null,
    spacing_source: scale?.source ?? null };
}
