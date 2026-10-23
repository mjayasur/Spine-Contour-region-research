import { predictionMatchesStudy } from '../renderer/data/predictions.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import { cervicalMeasurements, cervicalRows, cervicalMeasureGeometry, regionRunReason, regionKey, regionLabel, REGION_KEYS } from '../renderer/data/cervical.js';
import { landmarkAt, setLandmarkAt } from '../renderer/viewer/geometry.js';
import { constructionLabel } from '../renderer/viewer/canvas.js';
import { STORE_VERSION, validate } from '../renderer/data/persistence.js';
import { toCsv, toPairedCsv, parse } from '../renderer/data/csv.js';
import { parameterValues, measurementColumns, formatParameter } from '../renderer/data/parameters.js';
import { pairStudies } from '../renderer/data/pairing.js';
import { createMeasureQueue } from '../renderer/viewer/measure-queue.js';
import { planBatch } from '../renderer/data/batch.js';
import { loadWorkspaceStudies } from '../renderer/screens/workspace.js';
import { folderRows } from '../renderer/data/seeding.js';
import { segmentStudy, setFilePayload } from '../renderer/screens/analysis.js';
import { getState, setState } from '../renderer/store.js';

const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);
function study() {
  const s = { id: 'SP-9000', source: 'real', fileName: 'cervical.png', filePath: '/test/cervical.png',
    addedAt: '2026-09-23T12:00:00Z', view: 'Standing lateral', region: 'cervical', anteriorSide: 'left',
    subjectId: 'P1', timepoint: 'Pre-op', clinical: {},
    geometry: { region: 'cervical', anterior_side: 'left', c2_centroid: [140, 110],
      image_width: 500, image_height: 600, source_sha256: 'a'.repeat(64),
      vertebrae: { C2: { superior: null, inferior: [[100, 150], [200, 150]], quadrilateral: null },
        C7: { superior: [[120, 400], [180, 400]], inferior: [[120, 440], [180, 470]],
          quadrilateral: [[120, 400], [180, 400], [180, 470], [120, 440]] } },
      s1_superior: null, femoral_circles: [], hip_midpoint: null, l1_center: null },
    calibration: { version: 1, source_sha256: 'a'.repeat(64), width: 500, height: 600,
      coordinate_space: 'original_image', status: 'dicom', candidates: [], selected_index: null,
      spacing: { row_mm: .5, column_mm: .25, source: 'dicom_pixel_spacing' } },
  };
  s.measurements = cervicalMeasurements(s);
  return s;
}
const table = csv => parse(csv.split('\r\n').slice(2).join('\r\n'));

test('Cobb uses physical anisotropic directions; SVA is signed anterior-positive and uses only column spacing', () => {
  const s = study(), m = cervicalMeasurements(s);
  near(m.C2C7_COBB, 45); assert.equal(m.C2C7_SVA_PX, 40); assert.equal(m.C2C7_SVA_MM, 10);
  const mirrored = structuredClone(s);
  for (const v of Object.values(mirrored.geometry.vertebrae)) for (const plate of [v.superior, v.inferior])
    for (const p of plate ?? []) p[0] = 499 - p[0];
  mirrored.geometry.c2_centroid[0] = 499 - mirrored.geometry.c2_centroid[0];
  mirrored.geometry.anterior_side = 'right';
  const mirror = cervicalMeasurements(mirrored);
  near(mirror.C2C7_COBB, m.C2C7_COBB); assert.equal(mirror.C2C7_SVA_MM, m.C2C7_SVA_MM);
  s.geometry.c2_centroid[0] = 200; assert.equal(cervicalMeasurements(s).C2C7_SVA_PX, -20);
  s.geometry.c2_centroid[0] = 180; assert.equal(cervicalMeasurements(s).C2C7_SVA_MM, 0);
});

test('scale corrections, clears, source mismatch, and reload derive current values without reviving stale geometry spacing', () => {
  const s = study(), original = structuredClone(s);
  s.geometry.pixel_spacing = [.5, .25];
  const unscaled = { ...s, calibration: null };
  near(cervicalMeasurements(unscaled).C2C7_COBB, Math.atan(.5) * 180 / Math.PI);
  assert.equal(cervicalMeasurements(unscaled).C2C7_SVA_MM, null);
  assert.equal(cervicalRows(unscaled)[1].unit, 'px');
  s.calibration.spacing.column_mm = .5;
  assert.equal(cervicalMeasurements(s).C2C7_SVA_MM, 20);
  assert.deepEqual(cervicalMeasurements(JSON.parse(JSON.stringify(s))), cervicalMeasurements(s));
  s.calibration.source_sha256 = 'b'.repeat(64);
  assert.equal(cervicalMeasurements(s).C2C7_SVA_MM, null);
  assert.equal(cervicalMeasureGeometry(s.geometry, s.calibration).pixel_spacing, null);
  s.calibration = { ...original.calibration, width: 1000 };
  assert.equal(cervicalMeasurements(s).C2C7_SVA_MM, null);
});

test('missing/invalid landmarks remain absent, without lumbar fallthrough', () => {
  const s = study(); s.geometry.c2_centroid = null;
  assert.equal(cervicalMeasurements(s).C2C7_SVA_PX, null);
  assert.equal(cervicalRows(s)[1].absent, true);
  s.geometry.vertebrae.C2.inferior = null;
  assert.equal(cervicalMeasurements(s).C2C7_COBB, null);
  s.geometry.c2_centroid = [-1, 0];
  assert.equal(cervicalMeasurements(s).C2C7_SVA_PX, null);
  s.geometry.c2_centroid = [500, 100];
  assert.equal(cervicalMeasurements(s).C2C7_SVA_PX, null);
  assert.equal(constructionLabel(study().geometry, 'L1', study().measurements), null);
  assert.match(constructionLabel(study().geometry, 'C2C7_SVA', study().measurements).text, /10.0 mm/);
});

test('centroid edits preserve C2 null superior anatomy and stay inside the image', () => {
  const g = study().geometry;
  setLandmarkAt(g, 'C2', 'IA', [99, 150]);
  assert.equal(g.vertebrae.C2.quadrilateral, null);
  assert.equal(landmarkAt(g, 'C2', 'SA'), null);
  assert.deepEqual(g.vertebrae.C2.inferior[0], [99, 150]);
  setLandmarkAt(g, 'C2', 'CENTROID', [500, -5]);
  assert.deepEqual(g.c2_centroid, [499, 0]);
});

test('cervical region/orientation, partial landmarks, and null measurements survive persistence', () => {
  const s = study(); s.geometry.vertebrae.C7.inferior = null; s.measurements = cervicalMeasurements(s);
  const loaded = validate({ version: STORE_VERSION, studies: [JSON.parse(JSON.stringify(s))] })[0];
  assert.deepEqual(loaded.geometry, s.geometry); assert.deepEqual(loaded.measurements, s.measurements);
  assert.equal(loaded.region, 'cervical'); assert.equal(loaded.anteriorSide, 'left');
  assert.equal(loaded.measurements.C2C7_COBB, null);
});

test('current calibrated measurements agree across panel, parameter grid, CSV and paired CSV; uncalibrated mm stays blank', () => {
  const s = study();
  assert.equal(parameterValues(s).C2C7_SVA_MM, 10); assert.equal(parameterValues(s).PI, null);
  assert.ok(measurementColumns(false, true).some(c => c.key === 'C2C7_SVA_MM' && c.unit === 'mm'));
  assert.equal(formatParameter(10, 'mm'), '10.0mm');
  let csv = table(toCsv([s]));
  assert.equal(csv.rows[0]['C2-C7 SVA (mm)'], '10');
  assert.equal(csv.rows[0]['PI'], '');
  assert.equal(csv.rows[0]['Spine region'], 'Cervical');
  const post = { ...structuredClone(s), id: 'SP-9001', timepoint: 'Post-op', fileName: 'post.png' };
  post.geometry.c2_centroid[0] = 120;
  csv = table(toPairedCsv(pairStudies([s, post])));
  assert.equal(csv.rows[0]['Delta C2-C7 SVA (mm) Post-op'], '5');
  s.calibration = null;
  csv = table(toCsv([s]));
  assert.equal(csv.rows[0]['C2-C7 SVA (mm)'], '');
  assert.equal(csv.rows[0]['C2-C7 SVA (px)'], '40');
});

test('folder assignments are persisted per study and batch accepts cervical films with automatic orientation', () => {
  const files = ['/test/cervical/a.png', '/test/lumbar/b.png'];
  const rows = folderRows(files, '/test').map(row => ({ ...row, region: row.folder === 'cervical' ? 'cervical' : 'lumbar', anteriorSide: row.folder === 'cervical' ? 'right' : null }));
  const result = loadWorkspaceStudies({ studies: [], wsFiles: files, wsFolder: '/test', wsFolderRows: rows });
  assert.equal(result.studies[0].region, 'cervical'); assert.equal(result.studies[0].anteriorSide, 'right');
  assert.equal(result.studies[1].region, 'lumbar');
  const unconfirmed = { ...result.studies[0], anteriorSide: null };
  assert.equal(regionRunReason(unconfirmed), null);
  const plan = planBatch({ visible: [unconfirmed, result.studies[1]], selected: [], running: null });
  assert.deepEqual(plan.ids, [unconfirmed.id, result.studies[1].id]);
});

test('landmark recalculation sends cervical centroid, orientation and current scale then commits a complete pair', async () => {
  const s = study(); let state = { studies: [s], measurementDrafts: {} }, request;
  const queue = createMeasureQueue({ getState: () => state,
    setState: patch => { state = { ...state, ...patch(state) }; }, showToast: message => assert.fail(message), debounceMs: 1,
    measure: async geometry => { request = geometry; return { geometry, measurements: cervicalMeasurements({ ...s, geometry }) }; } });
  queue.replaceMeasured(s.id, s.geometry);
  const edited = structuredClone(s.geometry); setLandmarkAt(edited, 'C2', 'CENTROID', [145, 110]);
  queue.commitGeometry(s.id, edited);
  await new Promise(resolve => setTimeout(resolve, 20));
  assert.equal(request.region, 'cervical'); assert.equal(request.anterior_side, 'left');
  assert.deepEqual(request.c2_centroid, [145, 110]); assert.deepEqual(request.pixel_spacing, [.5, .25]);
  assert.equal(state.studies[0].measurements.C2C7_SVA_MM, 8.75);
  assert.deepEqual(state.measurementDrafts, {});
});

test('single/batch cervical run ignores saved manual orientation and requests anatomical inference', async () => {
  const saved = getState(); const oldWindow = globalThis.window; const s = { ...study(), measurements: null, geometry: null };
  let request;
  globalThis.window = { spineContour: { predict: async payload => { request = payload; throw new Error('intentional transport stop'); } } };
  try {
    setState({ studies: [s], running: null, batch: null, deletingStudies: false });
    setFilePayload(s.id, new Uint8Array([1]));
    const result = await segmentStudy(s.id, { batch: true });
    assert.equal(result.ok, false); assert.equal(request.bodyPart, 'cervical');
    assert.deepEqual(request.models, { vertebrae: 'cervical_slope_hrnet' }); assert.equal(request.anteriorSide, null);
  } finally { globalThis.window = oldWindow; setState(saved); }
});

test('sidecar hydration rejects old model, orientation, source, and failed-rerun snapshots while allowing manual edits', () => {
  const s = study(); const sidecar = { geometry: structuredClone(s.geometry), measurements: s.measurements };
  assert.equal(predictionMatchesStudy(s, sidecar), true);
  s.geometry.c2_centroid[0] += 3; // saved manual edit may differ from its original prediction
  assert.equal(predictionMatchesStudy(s, sidecar), true);
  assert.equal(predictionMatchesStudy({ ...s, geometry: null }, sidecar), false);
  assert.equal(predictionMatchesStudy(s, { ...sidecar, geometry: { ...sidecar.geometry, region: 'lumbar' } }), false);
  assert.equal(predictionMatchesStudy(s, { ...sidecar, geometry: { ...sidecar.geometry, anterior_side: 'right' } }), false);
  assert.equal(predictionMatchesStudy(s, { ...sidecar, geometry: { ...sidecar.geometry, source_sha256: 'b'.repeat(64) } }), false);
  s.predictionId = 'new-run';
  assert.equal(predictionMatchesStudy(s, sidecar), false);
  assert.equal(predictionMatchesStudy(s, { ...sidecar, prediction_id: 'old-run' }), false);
  assert.equal(predictionMatchesStudy(s, { ...sidecar, prediction_id: 'new-run' }), true);
  assert.equal(validate({ version: STORE_VERSION, studies: [s] })[0].predictionId, 'new-run');
});

test('persisted region/orientation disagreement discards a stale result without overriding explicit study setup', () => {
  const s = study(); s.anteriorSide = 'right';
  let loaded = validate({ version: STORE_VERSION, studies: [s] })[0];
  assert.equal(loaded.anteriorSide, 'right'); assert.equal(loaded.geometry, null); assert.equal(loaded.measurements, null);
  s.anteriorSide = 'left'; s.region = 'lumbar';
  loaded = validate({ version: STORE_VERSION, studies: [s] })[0];
  assert.equal(loaded.region, 'lumbar'); assert.equal(loaded.geometry, null); assert.equal(loaded.measurements, null);
});

test('regionKey: the region set by a person, else the one the run detected, else a lumbar result with none, else unknown', () => {
  assert.equal(regionKey({ region: 'cervical', geometry: { region: 'lumbar' } }), 'cervical');
  assert.equal(regionKey({ region: 'auto', geometry: { region: 'full_spine' } }), 'full_spine');
  assert.equal(regionKey({ region: 'auto', geometry: null, measurements: null }), 'unknown');
  assert.equal(regionKey({ region: 'auto', geometry: { region: 'bogus' }, measurements: { PI: 50 } }), 'lumbar', 'a result that recorded no valid region is an old lumbar one');
  assert.equal(regionKey({ source: 'demo', measurements: { PI: 50 } }), 'lumbar');
  assert.equal(regionKey({}), 'unknown');
  assert.equal(regionKey(null), 'unknown');
});
test('regionLabel reads the four labels and Unknown for anything else', () => {
  assert.deepEqual(REGION_KEYS.map(regionLabel), ['Unknown', 'Lumbar', 'Cervical', 'Full spine']);
  assert.equal(regionLabel('bogus'), 'Unknown');
});

// (2026-10-22, T1 slope) T1S is read through from the stored measurements, never recomputed here.
test('cervicalMeasurements reads T1S through; cervicalRows lists it third; absent is null', () => {
  const s = study();
  assert.equal(cervicalMeasurements(s).T1S, null, 'no stored T1S');
  s.measurements = { ...s.measurements, T1S: 25 };
  assert.equal(cervicalMeasurements(s).T1S, 25);
  const rows = cervicalRows(s);
  assert.deepEqual(rows.map(r => r.key), ['C2C7_COBB', 'C2C7_SVA', 'T1S']);
  assert.equal(rows[2].label, 'T1 SLOPE'); assert.equal(rows[2].value, 25); assert.equal(rows[2].unit, '\u00b0');
  assert.equal(rows[2].absent, false);
  assert.equal(cervicalRows(s, 'T1S')[2].highlight, true);
  assert.equal(cervicalRows(s, 'C2C7_COBB')[2].highlight, false);
  s.measurements = { ...s.measurements, T1S: 'x' };
  assert.equal(cervicalMeasurements(s).T1S, null, 'a non-finite value is absent');
  assert.equal(cervicalRows(s)[2].absent, true);
  assert.equal(cervicalMeasurements({ ...s, measurements: null }).T1S, null);
  assert.equal(cervicalMeasurements({ ...s, geometry: null, measurements: { T1S: 5 } }).T1S, 5);
});

// (Closing wave, F2) The T1 slope is cervical-family: a lumbar film's stored T1S is not read.
test('cervicalMeasurements reads no T1S for a lumbar study (F2)', () => {
  const lumbar = { region: 'lumbar', measurements: { T1S: 20 }, geometry: { region: 'lumbar' } };
  assert.equal(cervicalMeasurements(lumbar).T1S, null);
  assert.equal(cervicalMeasurements({ ...lumbar, region: 'cervical', geometry: { region: 'cervical' } }).T1S, 20);
});
