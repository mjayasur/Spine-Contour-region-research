import { cervicalMeasureGeometry } from '../data/cervical.js';
import { globalSvaMeasureGeometry } from '../data/global-sva.js';
import { debounce } from './interactions.js';

// The /measure round-trip, extracted from components/viewer.js so its bookkeeping is testable
// without a DOM: revisions are PER STUDY, the single debounce knows whose call it holds,
// committing on another study flushes the pending one instead of replacing it, replacing a
// study's geometry (a prediction, a reset) discards its pending or in-flight correction, and a
// failed round-trip discards its preview. Drafts live outside Study records:
// persistence and exports always see the last successfully measured pair. (2026-10-15, annotation editor round 2,
// section 4.3) A commit made with `clearFailure` (the level editor's, on a film no model produced) marks its study:
// the write that lands then nulls processingError and processingErrorAt in the same patch as the measurements, and
// a call that fails writes nothing, so the failure stays until a later flagged commit lands.
export function createMeasureQueue({ measure, getState, setState, showToast, debounceMs = 150 }) {
  const revisions = new Map(); // studyId -> latest revision issued
  const measured = new Map();  // studyId -> the geometry the study's current measurements describe
  const clearing = new Set();  // studyIds whose pending commit asked to clear the failure
  let pendingId = null;        // the study whose call sits in the debounce, or null

  function bump(studyId) {
    const next = (revisions.get(studyId) ?? 0) + 1;
    revisions.set(studyId, next);
    return next;
  }

  function writeStudy(studyId, patch) {
    setState((current) => {
      const measurementDrafts = { ...current.measurementDrafts };
      delete measurementDrafts[studyId];
      return {
        measurementDrafts,
        studies: current.studies.map((item) => (item.id === studyId ? { ...item, ...patch } : item)),
      };
    });
  }

  function discardDraft(studyId) {
    if (!getState().measurementDrafts?.[studyId]) return;
    setState((current) => {
      const measurementDrafts = { ...current.measurementDrafts };
      delete measurementDrafts[studyId];
      return { measurementDrafts };
    });
  }

  async function recalculate(studyId) {
    pendingId = null;
    const clear = clearing.delete(studyId);
    const revision = bump(studyId);
    const study = getState().studies.find((item) => item.id === studyId);
    const geometry = getState().measurementDrafts?.[studyId];
    if (!study || !geometry) { discardDraft(studyId); return; }
    try {
      const result = await measure(geometry.region === 'cervical'
        ? cervicalMeasureGeometry(geometry, study.calibration) : (geometry.region === 'full_spine' || geometry.pipeline_revision)
        ? globalSvaMeasureGeometry(geometry, study.calibration) : {
        vertebrae: geometry.vertebrae,
        s1_superior: geometry.s1_superior,
        femoral_circles: geometry.femoral_circles,
        t1_superior: geometry.t1_superior ?? null,
        t4_centroid: geometry.t4_centroid ?? null,
      });
      if (revision !== revisions.get(studyId)) return;
      // Legacy lumbar /measure responses omit region. Preserve Auto's resolved
      // type so an edit cannot turn a measured lumbar film back into an unknown film.
      if (geometry.region === 'lumbar' && !result.geometry.region) {
        result.geometry = { ...result.geometry, region: 'lumbar' };
      }
      const current = getState().studies.find((item) => item.id === studyId);
      if (!current || current.addedAt !== study.addedAt) { discardDraft(studyId); return; }
      measured.set(studyId, result.geometry);
      // The numbers a review was made over are being replaced, so the mark goes with them
      // (studies-table spec 2026-09-10, section 8.4). On the write, not derived: one line, one test.
      // The same write records manual-edit provenance in qc -- landmarks always, femoral when the circles moved.
      writeStudy(studyId, { measurements: result.measurements, geometry: result.geometry, reviewedAt: null,
        ...(clear ? { processingError: null, processingErrorAt: null } : {}),
        qc: { ...current.qc, ...(result.qc?.coverage ? { coverage: result.qc.coverage } : {}),
          manual_edits: { ...current.qc?.manual_edits, landmarks: true,
            femoral: Boolean(current.qc?.manual_edits?.femoral)
              || JSON.stringify(current.geometry?.femoral_circles) !== JSON.stringify(result.geometry.femoral_circles) } } });
    } catch (error) {
      if (revision !== revisions.get(studyId)) return;
      const known = measured.get(studyId);
      discardDraft(studyId);
      if (known) {
        showToast(`The correction was not applied — could not update measurements: ${error.message}`);
      } else {
        showToast(`Could not update measurements: ${error.message}`);
      }
    }
  }

  const schedule = debounce(recalculate, debounceMs);

  // Previews an edited geometry as a NEW reference and schedules the re-measure. Every edit
  // path ends here: drag release, keyboard nudge, retrace fit. The bump supersedes any response
  // still in flight for this study -- after a commit, none of them describes the store's geometry.
  // `clearFailure` rides the debounce: coalesced commits keep it once any of them set it.
  function commitGeometry(studyId, geometry, { clearFailure = false } = {}) {
    bump(studyId);
    if (!getState().studies.some((item) => item.id === studyId)) return;
    if (clearFailure) clearing.add(studyId);
    setState((current) => ({ measurementDrafts: { ...current.measurementDrafts, [studyId]: geometry } }));
    if (pendingId !== null && pendingId !== studyId) {
      schedule.cancel();
      recalculate(pendingId);
    }
    pendingId = studyId;
    schedule(studyId);
  }

  // Drops any correction pending or in flight for ONE study, and records the geometry its
  // measurements now describe (a fresh prediction, a reset).
  function replaceMeasured(studyId, geometry) {
    bump(studyId);
    clearing.delete(studyId);
    if (pendingId === studyId) {
      schedule.cancel();
      pendingId = null;
    }
    if (geometry) measured.set(studyId, geometry);
    else measured.delete(studyId);
    discardDraft(studyId);
  }

  return { commitGeometry, replaceMeasured };
}
