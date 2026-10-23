import { predictionMatchesStudy, isHandPlaced, hasStoredImage } from '../data/predictions.js';
import { studyRegion, requestedRegion, studyRegionLabel, requiresAnteriorSide, regionRunReason, validAnteriorSide, regionKey, regionLabel } from '../data/cervical.js';
import { imageConfidence, scorePercent } from '../data/confidence.js';
import { el, mount } from '../dom.js';
import { getState, setState, subscribe } from '../store.js';
import {
  predict, calibrate, saveCsv, savePrediction, loadPrediction, persistenceDisabledReason, readFile, selectFile, decodeFilm,
} from '../api.js';
import { showToast } from '../components/toast.js';
import { toCsv } from '../data/csv.js';
import { loadStudyImages, disposeStudyImages, thumbnailDataUri, bitmapFromBase64, imagesFromRaw } from '../viewer/canvas.js';
import { cropRaw, defaultWindow, rawFromDecoded } from '../viewer/window-level.js';
import { mountViewer, recordPrediction, forgetPrediction, getPrediction, commitGeometry, replaceMeasured } from '../components/viewer.js';
import { configureLevelEditor, forgetLevelEditor, requestExit, writeReviewRecord } from '../level-editor.js';
import { configureFilmWindow, forgetFilmWindow, windowNeedsDecode, redecodeKeptWindow } from '../film-window.js';
import { describeModels } from '../data/models.js';
import { describeProcessor, processorTitle } from '../data/processing.js';
import { WAIT_FOR_BATCH } from '../data/batch.js';
import { inferenceView, unsupportedViewReason } from '../data/inference-view.js';
import { studyName, defaultName } from '../data/labels.js';
import { displayStatus, isReviewed, reviewedLabel, reviewBlockedReason } from '../data/status.js';
import { statusBadge, unsupportedViewBadge } from '../components/status-badge.js';
import { failureReason, failureTitle } from '../data/failure.js';
import { mountMeasurements } from '../components/measurements.js';
import { mountLevelEditor } from '../components/level-editor.js';
import { mountClinicalData } from '../components/clinical-data.js';
import { calibrationForStudy } from '../calibration.js';
import { preferReviewedCalibration } from '../data/calibration.js';
import { findNeighbour, findPosition } from '../data/find.js';
import { FRESH_VIEW } from '../data/demo-visibility.js';

const BACK_SVG = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M19 12 H5"></path><path d="M11 6 L5 12 L11 18"></path></svg>';
// (2026-10-19, reviewer navigation) Previous and next, the same 24-unit stroke convention as BACK_SVG.
const PREV_SVG = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M15 5 L8 12 L15 19"></path></svg>';
const NEXT_SVG = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M9 5 L16 12 L9 19"></path></svg>';

// (2026-10-19) Which arrow goes where: Left and Up to the previous row of the Find list, Right and Down to the next.
export const NAV_STEP = Object.freeze({ ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1 });
// Controls whose own arrows win over navigation: text entry, dropdowns and sliders (the viewer's TEXT_ENTRY plus the
// window / level sliders and the tick boxes, which Space and the arrows reach through the browser). (2026-10-19, final
// review; ruling R14) Except the region bar's two selects, REGION_BAR_SELECT: handleNavKey says why.
const KEEPS_ARROWS = 'input, select, textarea, [contenteditable="true"]';
const REGION_BAR_SELECT = '.analysis-region-bar select';
// (2026-10-19, final review; ruling R15) How long a film must stay open before its original is previewed or decoded
// (render()'s film-loading chain): long enough for a held or repeated arrow key to pass over a film without loading it,
// short enough that a film opened to be looked at starts loading at once.
const FILM_SETTLE_MS = 150;

// (2026-10-19) The header's place in the Find list: `3 / 42`, or an en dash for a study the list's filters hide.
export function positionLabel({ index, count }) {
  return `${index === null ? '–' : index} / ${count}`;
}

// (2026-10-19) Opens the row before (`step` -1) or after (`step` 1) the open study in the Find list's order -- its
// search, its workspace, folder and status filters and its column sort (data/find.js findNeighbour) -- with the fresh
// view every opener gives a study (FRESH_VIEW, the data copy of screens/studies.js's). openId is a key of this screen's
// host (router.js ANALYSIS_KEYS), so the host remounts for the new film exactly as a click on the list does. Refused
// while editing: the arrows nudge handles there, and a switch would drop unsaved levels without the exit prompt. True
// when a study was opened.
export function openNeighbour(step) {
  const live = getState();
  if (live.screen !== 'analysis' || live.editing) return false;
  const next = findNeighbour(live, live.openId, step);
  if (!next) return false;
  setState({ openId: next.id, ...FRESH_VIEW });
  return true;
}

export function formatConfidence(qc) {
  return scorePercent(qc?.femoral?.confidence);
}

// (2026-10-01, issue #39; port spec 6) The Analysis header pill, on the list's rule
// (screens/studies.js buildRow): Unsupported view for a real, unmeasured film no model reads that is
// not running, else displayStatus with state.running and state.batch. `status` is what the pill shows
// and `badgeKey` is what update() rebuilds it on: the key carries the failure and its time, so a new
// failure under the same status rebuilds the pill and its tooltip. statusBadge never receives the key.
export function headerBadge(open, runningId = null, batch = null) {
  const unsupported = open.source === 'real' && open.measurements == null && runningId !== open.id && !inferenceView(open.view);
  const status = displayStatus(open, runningId, batch);
  const badgeKey = unsupported ? `unsupported:${open.view}` : `${status}|${open.processingErrorAt ?? ''}|${open.processingError ?? ''}`;
  return { unsupported, status, badgeKey };
}

// (2026-10-01, issue #39; port spec 6) The region note of a film whose header pill reads Failed, or
// null for every other film, whose note stays exactly what it was. `previewNote` is the preview
// message update() would otherwise show, already gated on its own rule; it follows the reason after
// a space.
export function failedRunNote(open, { unsupported, status }, previewNote = '') {
  if (unsupported || status !== 'fail') return null;
  const note = `Last run failed: ${open.processingError}`;
  return previewNote ? `${note} ${previewNote}` : note;
}

// (2026-10-20, issue #68) Whether the region bar's Region and Orientation selects may change the open film, and the
// tooltip that says why not. Before, ANY run or batch anywhere in the library disabled both, with no tooltip, so a
// reviewer stepping through a library while its batch ran over the rest could not change a film the run had labelled
// lumbar to cervical, and the Measurements panel kept the lumbar parameters. The change itself starts no run: it
// clears the film's result and sets the region (changeRegion), so only the film's OWN turn locks it -- its run in
// flight, or next up in the running batch, where the change would supersede the result the backend is about to
// return and the batch would count that turn as a failure. A film later in the queue is free: its turn reads the
// region afresh. A stopping batch starts no further turn. Edit mode locks both (ruling R21: the change wipes the
// levels under the editor), and a demo study has no film to run. `what` is 'region' or 'orientation', for the title.
export function regionLock(open, state, what = 'region') {
  if (!open || open.source !== 'real') return { locked: true, title: '' };
  if (state.editing) return { locked: true, title: `Finish editing before changing the ${what}` };
  const batch = state.batch;
  const turn = state.running === open.id || Boolean(batch && !batch.stopping && batch.ids.indexOf(open.id) === batch.done);
  if (turn) return { locked: true, title: `Wait for this film's segmentation to finish before changing the ${what}` };
  return { locked: false, title: '' };
}

// ---------------------------------------------------------------------------
// Transient per-study binary state. None of this belongs on the Study record --
// plan 05 persists state.studies to disk and validates its shape, so anything
// hung on the record ships. See BD-6 and BD-7.

// Raw file bytes, keyed by study id. screens/studies.js writes; segmentStudy reads.
// Transitional: plan 06 scans folders into studies that carry a filePath and no payload.
const filePayloads = new Map();

export function setFilePayload(studyId, data) {
  filePayloads.set(studyId, data);
}

// Exactly one study's decoded bitmaps, deliberately kept ACROSS navigation so that
// returning to an already-segmented study repaints without re-running /predict. Bounded
// to a single entry: ImageBitmaps are large. Without this, studies -> analysis ->
// studies -> analysis leaves a study that has measurements, therefore no run card, and
// no image -- a black stage with outlines floating on it.
let imageCache = null; // { studyId, images } | null

function cacheImages(studyId, images) {
  const outgoing = imageCache;
  imageCache = { studyId, images };
  // Never close bitmaps the live viewer is still drawing. The completion path deliberately
  // skips setImages for a study that is no longer on screen, which leaves that viewer
  // holding the OLD cache entry -- disposing it here would detach bitmaps it still draws,
  // and ctx.drawImage on a detached ImageBitmap throws inside a store subscriber, which
  // stops every subscriber after it. Same identity check as the hand-off gate.
  // (2026-10-10) ensureFilm replaces an entry with a copy that carries the decoded original and
  // SHARES the outgoing bitmaps; closing them would detach the copy's film.
  if (outgoing && outgoing.images !== images && outgoing.images.image !== images.image
      && !(mounted && mounted.studyId === outgoing.studyId)) {
    disposeStudyImages(outgoing.images);
  }
}

// The live mount, or null when this screen is not on screen.
let mounted = null;

// Every id-keyed cache this module holds for one study, dropped when the study is deleted.
// Ids are max+1, so the next film added can reuse a deleted id; without this the new record
// would inherit the old film's bytes and decoded bitmaps.
export function releaseStudy(studyId) {
  filePayloads.delete(studyId);
  runsByStudy.delete(studyId);
  // (2026-10-10) Its level editor too, open or stashed: a reused id must never inherit this film's levels.
  forgetLevelEditor(studyId);
  // (2026-10-11, gate feedback G3) And its window / level: the decoded original goes with the cache entry below.
  forgetFilmWindow(studyId);
  if (imageCache && imageCache.studyId === studyId) {
    // The entry always goes -- the next film can reuse this id. The bitmaps are CLOSED only
    // when no live viewer draws them (the same identity check cacheImages makes); a viewer
    // that still holds them keeps its own reference and repaints correctly until it detaches.
    if (!(mounted && mounted.studyId === studyId)) disposeStudyImages(imageCache.images);
    imageCache = null;
  }
}

let runRevision = 0;

// Runs started per study, for restoreFilm's guard (batch spec 8.3): a restore is dropped only when
// a run for the SAME study started meanwhile. runRevision above is global -- one run at a time --
// and a batch bumps it at every film's turn, which would drop every restore caught in that window
// and leave the card reading LOADING.
const runsByStudy = new Map();

// (2026-10-01, issue #39; port spec 4-5) The record keeps the plain sentence failureReason gives and
// the time, in one update; the toast and the batch outcome keep the raw message.
function withProcessingFailure(studies, studyId, addedAt, reason) {
  return studies.map((item) => item.id === studyId && item.addedAt === addedAt
    ? { ...item, processingError: failureReason(reason), processingErrorAt: new Date().toISOString() } : item);
}

function markProcessingFailure(studyId, addedAt, reason) {
  setState((state) => ({ studies: withProcessingFailure(state.studies, studyId, addedAt, reason) }));
}

// True while a relocate picker is open for a run that has not started. It refuses a second run
// (and so a second native dialog) WITHOUT claiming a segmentation is running -- the card must
// not say RUNNING while the app is waiting on a file dialog (no fabricated status).
let locating = false;

let restoreRevision = 0;

function currentStudy(state) {
  return state.studies.find((s) => s.id === state.openId) ?? null;
}

// Element-wise identity over a fixed-length key, the same shape screens/studies.js and
// components/viewer.js use. A null previous key never matches, so the first update() paints.
function sameConfidenceKey(a, b) {
  return a !== null && b !== null && a.length === b.length && a.every((v, i) => v === b[i]);
}

function teardown() {
  if (!mounted) return;
  mounted.viewer.detach();
  mounted.detachKeys();
  mounted = null;
  // imageCache deliberately survives -- that is the whole point of it.
}

// ---------------------------------------------------------------------------
// Subscribed ONCE, here at module scope, at import time. Not inside render().
//
// WHY NOT THE ROUTER (BD-2): this screen reads zoom/panX/panY/panMode, which change at
// pointermove rate. Adding them to router.js's SCREEN_KEYS would remount the screen host
// -- and therefore both <canvas> elements -- on every frame of a pan, orphaning the 2D
// contexts and stacking one set of listeners per frame. router.js:79-85 states this
// exception explicitly; its "add it here too" comment does not apply to this module.
//
// WHY NOT INSIDE render() (P2-3): render() runs on every navigation, so a subscription
// created there leaks one permanently-live listener per studies->analysis round trip,
// each rebuilding a detached tree on every later notification.
//
// ORDERING: this module's body evaluates before renderer/main.js's, because main.js
// imports router.js which imports this module, and store.js notifies listeners in
// insertion order. So on setState({screen: 'studies'}) the teardown below runs first and
// the router swaps the node second, which is the order that makes detach() correct.
subscribe((state) => {
  if (state.screen !== 'analysis') {
    teardown();
    return;
  }
  if (!mounted) return; // render() has not run for this navigation yet
  // (2026-10-19) Another study is opening (previous / next): the router remounts this screen for it in this same
  // notification, openId being one of its keys for this screen, so the outgoing mount is not updated with the
  // incoming study's record over the outgoing film.
  if (state.openId !== mounted.studyId) return;
  mounted.update();
});

// The film bytes: this session's payload, else the file at filePath, else null (moved or never had
// a path). `park` keeps what was read for a later re-run; a batch passes false (batch spec 8.3), so
// forty films' bytes are not held for the session.
async function filmBytes(study, { park = true } = {}) {
  const cached = filePayloads.get(study.id);
  if (cached) return cached;
  if (!study.filePath) return null;
  const bytes = await readFile(study.filePath);
  if (bytes && park) filePayloads.set(study.id, bytes);
  return bytes;
}

// Spec 13: a study whose source moved still lists with its numbers and opens; the moment the
// film is needed, offer to relocate it. Cancelling leaves the record untouched.
async function relocateFilm(study) {
  showToast(`${study.fileName} was not found. Choose its new location.`);
  const chosen = await selectFile();
  if (!chosen) return null;
  // IDENTITY, not existence, and checked here rather than only in the caller: this function
  // writes to the record on its own. The picker is modeless, so while it is open the study can
  // be deleted and a new film dropped -- and ids are max+1, so the new record can already carry
  // this id. Parking the chosen bytes and rewriting fileName/filePath by id alone would replace
  // the film the user just added. A reused id can never carry the deleted record's addedAt.
  const live = getState().studies.find((s) => s.id === study.id);
  if (!live || live.addedAt !== study.addedAt) return null;
  filePayloads.set(study.id, chosen.data);
  setState((state) => ({
    studies: state.studies.map((s) => (s.id === study.id ? { ...s, fileName: chosen.name, filePath: chosen.path } : s)),
  }));
  return chosen.data;
}

// The one path from a film to a committed record, for the viewer's button (interactive) and for a
// batch (batch spec 8.3). Returns an outcome and never throws. `state.running` is set and cleared
// here for both. In batch mode there is no relocate picker, no toast, no image-cache write for a
// study that is not on screen, and the bytes read from disk are not parked; every message the
// interactive path toasts becomes the outcome instead. Interactive callers ignore the outcome.
export async function segmentStudy(studyId, { batch = false } = {}) {
  // A relocate picker is already open for a run that has not started; a second click must not
  // raise a second native dialog. It is not a run, so it never claims one (interactive only: a
  // batch never opens the picker).
  if (!batch && locating) return { ok: false, reason: 'A file dialog is already open.' };
  // The !study return sits ABOVE the revision bump on purpose: bumping and then returning
  // early invalidates an in-flight run, whose completion would then return at a revision
  // check WITHOUT clearing `running` -- and every card would read RUNNING forever.
  const study = getState().studies.find((s) => s.id === studyId);
  if (!study) return { ok: false, reason: 'The study is no longer in the library.' };
  if (!inferenceView(study.view)) {
    const reason = unsupportedViewReason(study.view);
    markProcessingFailure(studyId, study.addedAt, reason);
    if (!batch) showToast(reason);
    return { ok: false, reason };
  }
  if (regionRunReason(study)) {
    const reason = regionRunReason(study);
    markProcessingFailure(studyId, study.addedAt, reason);
    if (!batch) showToast(reason);
    return { ok: false, reason };
  }
  // The record's identity, carried alongside its id for the checks after every await below.
  // Ids are max+1, so a deleted id is reused by the next film added; addedAt is not.
  const addedAt = study.addedAt;
  const revision = ++runRevision;
  runsByStudy.set(studyId, (runsByStudy.get(studyId) ?? 0) + 1);
  let data = null;
  let readError = null;
  if (!batch) locating = true;
  try {
    data = await filmBytes(study, { park: !batch });
    if (!data && !batch) data = await relocateFilm(study);
  } catch (error) {
    readError = error;
  } finally {
    if (!batch) locating = false;
  }
  if (readError) {
    const reason = `Could not read ${study.fileName}: ${readError.message}`;
    markProcessingFailure(studyId, addedAt, reason);
    if (!batch) showToast(reason);
    return { ok: false, reason };
  }
  // Spec 10: in a batch a missing film is a named failure. Keep any previous
  // measurements, mark the failed attempt, and let the user relocate the film.
  if (!data) {
    if (batch) markProcessingFailure(studyId, addedAt, 'file not found');
    return { ok: false, reason: 'file not found' };
  }
  // The picker is modeless and `locating` is this module's own, so a batch can have started while
  // it was open (batch spec 8.3). Running now would set `running` over the batch's id and put a
  // second /predict in flight. The record and the payload map already carry the relocated film;
  // the run itself waits for the user. ABOVE the revision check on purpose: the batch's own first
  // run has already moved runRevision, so the check below would refuse silently and the toast that
  // says why would never show. Only a batch can have started: `running` is set nowhere but here.
  if (!batch && getState().batch) {
    showToast(WAIT_FOR_BATCH);
    return { ok: false, reason: 'a batch is running' };
  }
  if (revision !== runRevision) return { ok: false, reason: 'superseded' };
  // After a relocation the record carries the NEW name; the `study` binding above is stale.
  // The filename matters: its extension drives the backend's decoder, so relocating a .jpg
  // to a .png has to send the new name with the new bytes.
  const current = getState().studies.find((s) => s.id === studyId);
  // Deleted while the bytes were being read or the relocate picker was open: nothing runs
  // for a record that is gone. The read above (filmBytes/relocateFilm) may have re-parked
  // the bytes under this id AFTER releaseStudy cleared them, so drop them again -- the next
  // film can reuse the id. runRevision is deliberately not bumped anywhere on delete.
  // `addedAt` as well as existence, because that reuse may already have happened: a record
  // with this id can be a DIFFERENT study, and running this study's bytes and measurements
  // onto it would silently replace the film the user just added.
  if (!current || current.addedAt !== addedAt) {
    filePayloads.delete(studyId);
    return { ok: false, reason: 'The study is no longer in the library.' };
  }

  // A bulk delete is clearing the library: it is about to remove records under these very ids,
  // so nothing new starts. Like every other refusal here this is an OUTCOME, never a bare
  // return -- a batch turn that got nothing back would resolve undefined and the driver would
  // read `ok` off it. Only the interactive path toasts.
  if (getState().deletingStudies) {
    const reason = 'Wait for study deletion to finish.';
    if (!batch) showToast(reason);
    return { ok: false, reason };
  }
  // The id, not a boolean: with a Studies list the user can open study B while A's /predict
  // is in flight, and the viewer and the list have to be able to ask WHICH study is running.
  // Every existing truthiness check still reads "a run is in flight" (one run at a time).
  // Metadata can change while file reading or relocation awaits.
  const view = inferenceView(current.view);
  if (!view) {
    const reason = unsupportedViewReason(current.view);
    markProcessingFailure(studyId, addedAt, reason);
    if (!batch) showToast(reason);
    return { ok: false, reason };
  }
  if (regionRunReason(current)) {
    const reason = regionRunReason(current);
    markProcessingFailure(studyId, addedAt, reason);
    if (!batch) showToast(reason);
    return { ok: false, reason };
  }
  const requestId = crypto.randomUUID();
  setState({ running: studyId, runStage: { requestId, mode: getState().performance.mode,
    stage: 'starting', message: 'Sending the image to the processing worker', elapsed_seconds: 0 } });
  let warning = null;
  try {
    const response = await predict({
      requestId,
      name: current.fileName,
      data,
      modality: 'xray',
      bodyPart: requestedRegion(current),
      anteriorSide: studyRegion(current) === 'cervical' ? null : current.anteriorSide ?? null,
      view,
      ...(requestedRegion(current) === 'auto' ? {} : { models: requestedRegion(current) === 'cervical'
        ? { vertebrae: 'cervical_slope_hrnet' } : requestedRegion(current) === 'full_spine'
        ? { vertebrae: 'dual_hrnet' } : getState().models }),
      calibration: calibrationForStudy(current),
    });
    if (revision !== runRevision) return { ok: false, reason: 'superseded' };

    if (getState().runStage?.requestId === requestId && getState().runStage.cancelling) {
      throw new Error('Processing cancelled.');
    }
    setState({ runStage: { ...getState().runStage, message: 'Preparing the viewer and saving results',
      stage: 'saving', completed: null, total: null } });
    const images = await loadStudyImages(response);
    if (revision !== runRevision) {
      disposeStudyImages(images);
      return { ok: false, reason: 'superseded' };
    }

    // The record's identity again, before any side effect (batch spec 11): deleteStudy refuses
    // while `running` is this id, so today nothing can remove the record between the check above
    // and here -- this guards a future removal path. A vanished record gets no sidecar, no cache
    // entry, no prediction snapshot and no commit; `running` was set for it and is cleared here.
    const stillHere = getState().studies.find((s) => s.id === studyId);
    if (!stillHere || stillHere.addedAt !== addedAt) {
      disposeStudyImages(images);
      setState({ running: null, runStage: null });
      return { ok: false, reason: 'The study is no longer in the library.' };
    }

    const thumbnail = thumbnailDataUri(images.image);
    response.prediction_id = requestId;
    response.calibration = preferReviewedCalibration(response.calibration, calibrationForStudy(stillHere));

    // The sidecar first, then the record: a record that says "segmented" must point at a film
    // that exists. A failed sidecar write is reported and the run still completes — the study
    // opens to FILM UNAVAILABLE next time, and a re-run recreates it. Neither toast starts with
    // "Could not": tools/smoke/run-and-wait.js treats that prefix as a failed run. In a batch the
    // persistence notice was raised once at the start, and the sidecar failure is the outcome's
    // warning, one clause in the closing toast.
    if (persistenceDisabledReason()) {
      if (!batch) showToast('Studies are not being saved this session, so the segmentation images were not stored.');
    } else {
      try {
        await savePrediction(studyId, response);
      } catch (error) {
        warning = `the segmentation images could not be stored: ${error.message}`;
        if (!batch) showToast(`Saved the measurements, but ${warning}`);
      }
    }
    if (revision !== runRevision) { disposeStudyImages(images); return { ok: false, reason: 'superseded' }; }

    // ORDER MATTERS (BD-6). setState notifies synchronously, so the module-scope
    // subscription's update() runs INSIDE the setState call below and asks the viewer to
    // repaint. The images have to be in place first, or that first paint sizes nothing
    // and draws nothing: every measurement populates while the stage stays black until
    // an unrelated click happens to fire the next update.
    //
    // Hand off to the live viewer only if it is still showing the study this run was
    // for. The user may have navigated to a different study (or back to Studies) while
    // /predict was in flight -- runRevision only guards against a SECOND run for the
    // SAME study, not against navigation, so without this check a slow-resolving run for
    // A can paint A's bitmaps into B's live viewer. This is the completion-time sibling
    // of the re-hand guard in render() (`imageCache.studyId === study.id`): that one checks
    // identity before handing a freshly mounted viewer its cached bitmaps, this one
    // checks identity before handing a freshly resolved run its live viewer. The cache
    // write and the setState below stay unconditional for an interactive run -- A's results are
    // real and belong in the store regardless of what's on screen; only the live paint is gated.
    // A BATCH writes the single-entry cache only for the study on screen (batch spec 8.3): forty
    // films must not evict the open study's bitmaps, and the decoded bitmaps of a film nobody is
    // looking at have served their thumbnail and are closed here.
    const onScreen = Boolean(mounted && mounted.studyId === studyId);
    if (!batch || onScreen) cacheImages(studyId, images);
    if (onScreen) mounted.viewer.setImages(images);
    else if (batch) {
      disposeStudyImages(images);
      // This study's OWN cache entry predates the result: an original-film preview (previewOriginal),
      // a film decoded from its original (ensureFilm) or an earlier run's images. Left in place, the
      // next open re-hands it instead of restoring the new sidecar (render()'s needsRestore is false
      // while an entry exists) and ensureFilm returns it as this result's film. Dropped and closed
      // here; no viewer draws it, since the study is not on screen (cacheImages' identity check).
      if (imageCache && imageCache.studyId === studyId && !(mounted && mounted.studyId === studyId)) {
        const stale = imageCache.images;
        imageCache = null;
        disposeStudyImages(stale);
      }
    }

    recordPrediction(studyId, response);

    // The finished study's own edit mode ends -- its geometry was just replaced under the
    // handles -- but ONLY if it is the study on screen. `editing` and `selection` belong to
    // `openId` (every writer of openId resets both, screens/studies.js FRESH_VIEW), and with a
    // Studies list the user may have opened study B and entered edit mode while A's /predict
    // was in flight: the viewer disables Edit only for the running study itself. A's completion
    // must not drop B out of edit mode. The error path below never touched either key.
    // (2026-10-10) The study's level editor goes with its edit mode, and forgetLevelEditor below drops a
    // stashed one: the levels it holds describe geometry this result has just replaced.
    setState((state) => ({
      running: null,
      runStage: null,
      editing: state.openId === studyId ? false : state.editing,
      selection: state.openId === studyId ? null : state.selection,
      levelEditor: state.levelEditor?.studyId === studyId ? null : state.levelEditor,
      studies: state.studies.map((s) => (s.id === studyId
        ? { ...s, anteriorSide: response.geometry?.region === 'cervical' ? null : s.anteriorSide, predictionId: requestId, measurements: response.measurements, geometry: response.geometry, qc: response.qc ?? null,
          calibration: preferReviewedCalibration(response.calibration, calibrationForStudy(s)), thumbnail,
          // A re-run replaces every number a review was made over (studies-table spec 2026-09-10, section 8.4, site 1).
          reviewedAt: null, processingError: null, processingErrorAt: null }
        : s)),
    }));
    forgetLevelEditor(studyId);
    // (2026-10-11, gate feedback G3) The result's images carry no decoded original: the window goes with the old
    // film, and the next W/L press or right-button drag decodes again.
    forgetFilmWindow(studyId);
    return warning ? { ok: true, warning } : { ok: true };
  } catch (error) {
    if (revision === runRevision) {
      if (error.message === 'Processing cancelled.') {
        setState({ running: null, runStage: null });
        if (!batch) showToast('Processing cancelled. Previous results were kept.');
        return { skipped: true, cancelled: true };
      }
      setState((state) => ({ running: null, runStage: null,
        studies: withProcessingFailure(state.studies, studyId, addedAt, error.message) }));
      if (!batch) showToast(`Could not segment: ${error.message}`);
      return { ok: false, reason: error.message };
    }
    return { ok: false, reason: 'superseded' };
  }
}

// A persisted study opened after a restart has numbers but no bitmaps. Read its sidecar,
// decode, hand the bitmaps to the live viewer, and re-record the prediction snapshot with the
// STORED geometry as the measured one. Guarded like segmentStudy but per study: a newer
// restore, a run for THIS study, or its deletion drops this one's result; navigation does not,
// and `imageCache` deliberately survives it.
// Resolves to its outcome, which render() ignores and ensureFilm reads: {status: 'restored', images},
// {status: 'superseded'}, {status: 'missing'} (no sidecar for this result) or {status: 'failed', reason}.
async function restoreFilm(studyId) {
  const revision = ++restoreRevision;
  // Per study, not runRevision: a batch bumps the global counter at every film's turn (batch spec
  // 8.3), and this restore is only stale if THIS study started a run meanwhile.
  const runAtStart = runsByStudy.get(studyId) ?? 0;
  const runMoved = () => (runsByStudy.get(studyId) ?? 0) !== runAtStart;
  // The record's identity at the start, checked again below. `undefined` cannot reach here
  // (the caller restores the OPEN study), and a null placeholder still fails the comparison
  // against any real record, which is the safe direction.
  const addedAt = getState().studies.find((s) => s.id === studyId)?.addedAt ?? null;
  const live = () => mounted && mounted.studyId === studyId;
  if (live()) mounted.viewer.setFilmStatus('loading');
  try {
    // The sidecar READ is gated exactly like the writes. When the store on disk was refused
    // (a newer version, a broken record) persistence is off for the session and nextId()
    // restarts at SP-1000 over a library this build cannot see -- so predictions/SP-1000.json
    // is the PREVIOUS library's sidecar: another patient's radiograph, mask and geometry, and
    // recordPrediction would make them this study's RESET TO PREDICTION target. Treat it as a
    // missing sidecar, which already has a defined outcome (FILM UNAVAILABLE; a re-run
    // recreates it) rather than a wrong one.
    const sidecar = persistenceDisabledReason() ? null : await loadPrediction(studyId);
    if (revision !== restoreRevision || runMoved()) return { status: 'superseded' };
    if (!predictionMatchesStudy(getState().studies.find(s => s.id === studyId), sidecar)) {
      if (live()) mounted.viewer.setFilmStatus('missing');
      return { status: 'missing' };
    }
    const images = await loadStudyImages(sidecar);
    const study = getState().studies.find((s) => s.id === studyId);
    // Deleted while the sidecar was being read and decoded: releaseStudy has already cleared
    // this id's caches, and nothing may be re-parked under an id the next film can reuse --
    // cacheImages + recordPrediction would restore the film, the snapshot AND the measured
    // geometry under a dead id, and the reusing record would open on the deleted study's film
    // with RESET TO PREDICTION live over its numbers. Existence is not enough: ids are max+1,
    // so a record with this id may be the film added AFTER the delete. Identity is `addedAt`,
    // which a reused id never carries.
    if (!study || study.addedAt !== addedAt || revision !== restoreRevision || runMoved()
        || !predictionMatchesStudy(study, sidecar)) {
      disposeStudyImages(images);
      return { status: 'superseded' };
    }
    cacheImages(studyId, images);
    recordPrediction(studyId, sidecar, study.geometry ? study.geometry : sidecar.geometry);
    if (live()) {
      mounted.viewer.setFilmStatus(null);
      mounted.viewer.setImages(images);
      mounted.update();
    }
    return { status: 'restored', images };
  } catch (error) {
    if (revision !== restoreRevision) return { status: 'superseded' };
    if (live()) mounted.viewer.setFilmStatus('missing');
    showToast(`Could not load the film for ${studyId}: ${error.message}`);
    return { status: 'failed', reason: error.message };
  }
}

// ---------------------------------------------------------------------------
// (2026-10-10, level editor, Task 8) The film for the stage and its decoded original for the
// display-only window / level (plan D1, D11). The decoded original lives on the cached images
// object as `images.raw`, never in the store; releaseStudy drops it with the cache entry.

const FILM_NOT_FOUND = 'The source film could not be found.';
const FILM_MISMATCH = 'The original film does not match the stored image.';
const SAVED_FILM_MISSING = 'The saved segmentation was not found.';

function filmOutcome(images, raw, reason, superseded = false) {
  return { images, raw, reason, superseded };
}

function cachedImages(studyId) {
  return imageCache && imageCache.studyId === studyId ? imageCache.images : null;
}

// The ensureFilm in flight, so two callers (entering edit mode, then opening the W/L panel) share one
// decode. Keyed by the record's identity and its run count, like every guard below.
let filmLoad = null; // { studyId, addedAt, runs, promise } | null

/**
 * The open study's images and its decoded original. Never throws.
 *
 * `images`: the cache entry; else, for a study with a model result (not isHandPlaced), its restored sidecar
 * (restoreFilm, every one of its guards kept); else -- a film with no result, the main case, or one whose
 * levels were placed by hand with no model run (R16, isHandPlaced) -- built from the decoded original at its default
 * window, cached and handed to the live viewer so the stage shows the film.
 * `raw`: the RawFilm, decoded at most once per study and cached on `images.raw` (a NEW images object,
 * so the viewer's reference-keyed static gate repaints); cropped to a toolbar-cropped stored image's
 * height, refused on a width mismatch.
 *
 * @param {string} studyId
 * @returns {Promise<{images: object|null, raw: import('../viewer/window-level.js').RawFilm|null,
 *   reason: string|null, superseded: boolean}>}
 *   `reason`: display-ready, set when `raw` is null and the call was not superseded; null otherwise.
 *   `superseded`: true when the study was deleted, re-run, had its Region changed or its cache entry
 *   replaced while this call was in flight; then `images`, `raw` and `reason` are null and nothing was
 *   written. Not a failure to show: the caller may simply call again.
 */
export async function ensureFilm(studyId) {
  const study = getState().studies.find((s) => s.id === studyId);
  const runs = runsByStudy.get(studyId) ?? 0;
  if (study && filmLoad && filmLoad.studyId === studyId && filmLoad.addedAt === study.addedAt && filmLoad.runs === runs) {
    return filmLoad.promise;
  }
  const promise = loadFilm(studyId).catch((error) => filmOutcome(null, null, error?.message || 'The original film could not be loaded.'));
  const entry = { studyId, addedAt: study?.addedAt, runs, promise };
  filmLoad = entry;
  promise.then(() => { if (filmLoad === entry) filmLoad = null; });
  return promise;
}

async function loadFilm(studyId) {
  const start = getState().studies.find((s) => s.id === studyId);
  if (!start) return filmOutcome(null, null, 'The study is no longer in the library.');
  // The record's identity and its run count at the start, checked after every await below: ids are
  // max+1, so a deleted id is reused by the next film, and a run or a Region change (changeRegion
  // bumps the count too) replaces the film this call is resolving.
  const addedAt = start.addedAt;
  const runAtStart = runsByStudy.get(studyId) ?? 0;
  const current = () => {
    const study = getState().studies.find((s) => s.id === studyId);
    return study && study.addedAt === addedAt && (runsByStudy.get(studyId) ?? 0) === runAtStart ? study : null;
  };
  const superseded = () => filmOutcome(null, null, null, true);

  // (1) The stored image. Only a model run writes one (a sidecar). A study whose geometry came from the level
  // editor alone (isHandPlaced: no prediction id and no qc.models, ruling R16 as amended) has none: it is a film
  // with no result here, shown from its decoded original, never FILM UNAVAILABLE. R-a below still holds for every
  // model result that lost its sidecar, a pre-v1.0.10 one (qc.models, no id) included. (2026-10-12) The test is
  // data/predictions.js hasStoredImage, which imageInfo's stored-only read shares (R42).
  let images = cachedImages(studyId);
  if (!images && hasStoredImage(start)) {
    const restored = await restoreFilm(studyId);
    if (!current() || restored.status === 'superseded') return superseded();
    // The result's geometry is in the stored image's pixels and FILM UNAVAILABLE (re-run to restore
    // it) is that film's defined outcome: the original is not swapped in under it.
    if (restored.status !== 'restored') return filmOutcome(null, null, restored.reason || SAVED_FILM_MISSING);
    images = restored.images;
  }
  // The cache must still hold what this call resolved: a replaced entry may already be closed.
  const unchanged = () => current() && cachedImages(studyId) === images;
  if (!unchanged()) return superseded();
  if (images?.raw) return filmOutcome(images, images.raw, null);

  // (2) The decoded original. The bytes are not parked: the decode is cached instead, and a read that
  // lands after a delete must not leave bytes under an id the next film can reuse.
  const study = current();
  let raw;
  try {
    const data = await filmBytes(study, { park: false });
    if (!unchanged()) return superseded();
    if (!data) return filmOutcome(images, null, FILM_NOT_FOUND);
    raw = rawFromDecoded(await decodeFilm({ name: study.fileName, data }));
    if (!unchanged()) return superseded();
    if (images && raw.width !== images.width) return filmOutcome(images, null, FILM_MISMATCH);
    if (images) raw = cropRaw(raw, images.height);
  } catch (error) {
    if (!unchanged()) return superseded();
    return filmOutcome(images, null, error?.message || 'The original film could not be decoded.');
  }

  // (3) Attach. A new images object either way; the cached one is never changed in place.
  const next = images ? { ...images, raw } : imagesFromRaw(raw, defaultWindow(raw));
  cacheImages(studyId, next);
  if (mounted && mounted.studyId === studyId) {
    mounted.viewer.setImages(next);
    mounted.update();
  }
  return filmOutcome(next, raw, null);
}

/**
 * (2026-10-10, level editor) What the label record says about the stored image (plan Task 10): the film's file
 * name, the stored image's size and the hex SHA-256 of its grayscale bytes -- the red channel of every pixel,
 * row by row -- which ties a record to the exact pixels with no identifier. For a film with no result the
 * "stored image" is the decoded original at its default window, the image every point was placed on.
 * `maskPixels` is the label mask's RGBA (red channel = label id) read from the cached mask bitmap, and `labels` the
 * prediction's `{ L1: 20, ... }` map, for the record's contours (data/contours.js); both are null for a film shown
 * from its decoded original, which has no mask. Rejects with a display-ready message when there is no film.
 *
 * (2026-10-12, review round 1 of gate feedback G6; ruling R42) `storedOnly`, the label record Mark reviewed writes:
 * for a model result (data/predictions.js hasStoredImage) the stored image is read without its decoded original, which
 * the record never needs: the cache entry when it holds this film, read at once with no await before its pixels, else
 * the study's own sidecar read for this call alone (readStoredImages). No /decode runs, nothing is cached, no viewer is
 * handed anything and no restore is started or superseded, so another film opened straight after the click can
 * neither take the image from under the read nor be held up by it. The facts are the ones the full read gives: the
 * decode only ever adds `raw` beside the same stored image, mask and labels (loadFilm step 3). A film whose stored
 * image is its decoded original (no result, or placed by hand, R16) still goes through ensureFilm.
 * `sidecarGeometry` is the model's own geometry from the sidecar such a private read loaded, else null: a restore
 * records it as the viewer's prediction snapshot, which a private read does not do, so a caller that needs the model's
 * geometry has it even when no restore ever lands (another film opened before it did).
 * @param {string} studyId
 * @param {{storedOnly?: boolean}} [options]
 * @returns {Promise<{file: string, sha256: string, width: number, height: number,
 *   maskPixels: Uint8ClampedArray|null, labels: object|null, sidecarGeometry: object|null}>}
 */
export async function imageInfo(studyId, { storedOnly = false } = {}) {
  const start = getState().studies.find((s) => s.id === studyId);
  let images;
  let own = null; // images read for this call alone (readStoredImages): closed once their pixels are read
  let sidecarGeometry = null;
  if (storedOnly && hasStoredImage(start)) {
    images = cachedImages(studyId);
    if (!images) {
      const read = await readStoredImages(start);
      images = own = read.images;
      sidecarGeometry = read.geometry;
    }
  } else {
    const outcome = await ensureFilm(studyId);
    if (outcome.superseded) throw new Error('The film changed while it was being read. Try again.');
    if (!outcome.images) throw new Error(outcome.reason || SAVED_FILM_MISSING);
    images = outcome.images;
  }
  try {
    const study = getState().studies.find((s) => s.id === studyId);
    if (!study) throw new Error('The study is no longer in the library.');
    const { width, height } = images;
    const scratch = document.createElement('canvas');
    scratch.width = width;
    scratch.height = height;
    const context = scratch.getContext('2d', { willReadFrequently: true });
    context.drawImage(images.image, 0, 0);
    const rgba = context.getImageData(0, 0, width, height).data;
    const grey = new Uint8Array(width * height);
    for (let i = 0, o = 0; i < grey.length; i += 1, o += 4) grey[i] = rgba[o];
    // The mask is read before the digest's await, while the cached bitmap is certainly open.
    let maskPixels = null;
    if (images.mask) {
      context.clearRect(0, 0, width, height);
      context.drawImage(images.mask, 0, 0);
      maskPixels = context.getImageData(0, 0, width, height).data;
    }
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', grey));
    const sha256 = Array.from(digest, (byte) => byte.toString(16).padStart(2, '0')).join('');
    return { file: study.fileName, sha256, width, height, maskPixels, labels: images.labels ?? null, sidecarGeometry };
  } finally {
    disposeStudyImages(own);
  }
}

// (2026-10-12, R42) A model result's stored image read from its sidecar for one caller alone (imageInfo's stored-only
// read when the cache does not hold the film, as when Mark reviewed is clicked before the restore has put it on the
// stage), with the sidecar's geometry, the model's own: restoreFilm's two sidecar guards (persistence off reads as no
// sidecar; the sidecar must belong to this record's result) and none of its side effects. The caller closes the images.
async function readStoredImages(study) {
  const sidecar = persistenceDisabledReason() ? null : await loadPrediction(study.id);
  const live = getState().studies.find((s) => s.id === study.id);
  if (!live || live.addedAt !== study.addedAt) throw new Error('The study is no longer in the library.');
  if (!predictionMatchesStudy(live, sidecar)) throw new Error(SAVED_FILM_MISSING);
  return { images: await loadStudyImages(sidecar), geometry: sidecar.geometry };
}

// (The final review of round 2, I1) The level editor's way to take a study's result away: a film no model produced
// whose levels were all taken off again (renderer/level-editor.js sync). Whatever /measure it has pending or in flight
// is orphaned first (the queue's replaceMeasured, which also drops its draft), so no late answer lands over the patch;
// then the patch is written in one update, onto the record that was asked about (`addedAt`).
function clearResult(studyId, patch) {
  const study = getState().studies.find((s) => s.id === studyId);
  replaceMeasured(studyId, null);
  if (!study) return;
  setState((state) => ({
    studies: state.studies.map((s) => (s.id === studyId && s.addedAt === study.addedAt ? { ...s, ...patch } : s)),
  }));
}

// The level editor's wiring (renderer/level-editor.js) gets the viewer's measure queue and prediction snapshots,
// and this screen's film loader, image facts and result clear. Once, at import, like the subscription above.
configureLevelEditor({ commitGeometry, clearResult, getPrediction, ensureFilm, imageInfo });
// (2026-10-11, gate feedback G3) The view's window / level (renderer/film-window.js) decodes through the same loader.
configureFilmWindow({ ensureFilm });


let previewRevision = 0;
// A clean original-image preview (including DICOM) gives the user evidence for the
// required anterior-side selection. This is never a prediction or a saved measurement.
async function previewOriginal(studyId) {
  const start = getState().studies.find(s => s.id === studyId);
  if (!start || start.geometry || !requiresAnteriorSide(start)) return;
  const revision = ++previewRevision;
  const runAtStart = runsByStudy.get(studyId) ?? 0;
  const live = () => {
    const current = getState().studies.find(s => s.id === studyId);
    return mounted?.studyId === studyId && getState().screen === 'analysis'
      && revision === previewRevision && (runsByStudy.get(studyId) ?? 0) === runAtStart
      && current?.addedAt === start.addedAt && !current.geometry
      && requiresAnteriorSide(current) && getState().running !== studyId;
  };
  if (!live()) return;
  mounted.previewMessage = 'Loading original radiograph…';
  mounted.update();
  try {
    // (2026-10-19, final review; ruling R15) Not parked: opening a film is not a run, and arrow-stepping through
    // hundreds of unsegmented films (every new film is Auto detect, which previews here) parked every file in
    // filePayloads for the session, released only on delete. A later run re-reads the film from its filePath, as a
    // batch does; a film dropped with no path was parked when it was added and is found there.
    const data = await filmBytes(start, { park: false });
    if (!live()) return;
    if (!data) throw new Error('The source film could not be found.');
    const response = await calibrate({ name: start.fileName, data, previewOnly: true });
    if (!live()) return;
    const image = await bitmapFromBase64(response.image_png);
    if (!live()) { image.close(); return; }
    const images = { image, mask: null, overlayCanvas: null, width: image.width, height: image.height, preview: true };
    const outgoing = imageCache?.studyId === studyId ? imageCache.images : null;
    cacheImages(studyId, images);
    mounted.viewer.setImages(images);
    if (outgoing && outgoing !== images) disposeStudyImages(outgoing);
    mounted.previewMessage = '';
    mounted.update();
  } catch (error) {
    if (!live()) return;
    mounted.previewMessage = `Original preview unavailable: ${error.message}`;
    mounted.update();
  }
}

// (2026-10-19) A film with no result shows its X-ray by default: the decoded original at its default window (ensureFilm,
// the level editor's loader, which hands the images to the live viewer itself). (2026-10-19, final review) Only a lumbar
// film comes here (requestedRegion lumbar, or an Auto detect run that found lumbar): an Auto detect, cervical or full-spine
// film takes previewOriginal (loadOriginal, render()), as it did before, so the UNSEGMENTED strip is the lumbar film's. The
// region note reads the progress and, when the original cannot be read, why; the UNSEGMENTED card then stays as it
// was, its Run segmentation offering the relocate picker. Guarded like previewOriginal: nothing is written to a mount
// that has moved on. Never throws: ensureFilm resolves an outcome.
async function showOriginal(studyId) {
  const live = () => mounted?.studyId === studyId && getState().screen === 'analysis';
  if (!live()) return;
  mounted.previewMessage = 'Loading original radiograph…';
  mounted.update();
  const outcome = await ensureFilm(studyId);
  if (!live()) return;
  mounted.previewMessage = outcome.superseded || outcome.images ? '' : `Original preview unavailable: ${outcome.reason}`;
  mounted.update();
}

// (2026-10-19) The original for a real film with no result: cervical and full-spine films keep previewOriginal (the
// backend's clean preview, the evidence for the anterior-side choice, with its own guards); every other film is
// decoded here (showOriginal). Nothing for a demo study, which has no film, or a film with a result.
function loadOriginal(studyId) {
  const study = getState().studies.find((s) => s.id === studyId);
  if (!study || study.source !== 'real' || study.geometry) return;
  if (requiresAnteriorSide(study)) previewOriginal(studyId);
  else showOriginal(studyId);
}

export function render(state) {
  // A re-render (the router calls this whenever SCREEN_KEYS change) must not leave the
  // previous mount's listeners live.
  teardown();

  const study = currentStudy(state);
  if (!study) {
    // Still not reachable, but no longer for the old reason: the sidebar's OPEN STUDY card
    // IS an Analysis entry point now, and it sets `screen` without touching openId. It is
    // safe because the card only renders while the study is in state.studies
    // (components/sidebar.js's openStudyCard), and `studies` is in the router's SIDEBAR_KEYS,
    // so deleting the open study remounts the sidebar and the card goes with it -- the delete
    // path in screens/studies.js clears openId and returns to the list in the same setState.
    // Rendering a header-and-empty-viewer shell over a study that isn't there would still be
    // worse than one extra branch.
    return el('main', { class: 'placeholder-screen' }, el('p', {}, 'No study is open.'));
  }

  const backButton = el('button', {
    type: 'button',
    class: 'icon-btn',
    title: 'Back to studies',
    'aria-label': 'Back to studies',
    innerHTML: BACK_SVG,
    // (2026-10-10) While editing, Back asks first when level edits are unsaved (D7), then returns to the list.
    onClick: () => {
      if (getState().editing) requestExit({ then: 'back' });
      else setState({ screen: 'studies', editing: false, selection: null });
    },
  });

  // The name is an input, not text: it is the one field on the header the user owns. It defaults
  // to the film's filename and is theirs to change. The SP-nnnn id underneath never moves -- it
  // names the sidecar, keys the delete and is the CSV's Study ID -- so a rename is cosmetic by
  // construction and can never orphan a file.
  const nameField = el('input', {
    type: 'text',
    class: 'analysis-name',
    'aria-label': 'Study name',
    // A boolean, not the string 'false': dom.js's el() assigns to the property when the key
    // exists on the node, and the non-empty string 'false' would coerce to true.
    spellcheck: false,
    onKeydown: (event) => {
      if (event.key === 'Enter') { event.preventDefault(); nameField.blur(); return; }
      if (event.key === 'Escape') {
        event.preventDefault();
        const open = currentStudy(getState());
        if (open) nameField.value = studyName(open);
        nameField.blur();
      }
    },
    // Deferred: Chromium fires this while the node may still be being replaced, and setState
    // notifies synchronously. The clinical drawer defers its writes for the same reason.
    onBlur: () => queueMicrotask(commitName),
  });
  const headerMeta = el('div', { class: 'analysis-meta' });
  const confidenceValue = el('span', { class: 'confidence-value' });

  // Blank means "go back to the film's own name", not "no name": an empty header would leave the
  // user with nothing to recognise the study by, and the filename is always recoverable.
  function commitName() {
    const open = currentStudy(getState());
    if (!open || open.source === 'demo') return;
    const typed = nameField.value.trim();
    const next = typed === '' ? defaultName(open.fileName) : typed;
    if ((open.name ?? null) === (next ?? null)) return;
    // A new record and a new array: the store's records are never edited in place.
    setState((state) => ({
      studies: state.studies.map((study) => (study.id === open.id ? { ...study, name: next } : study)),
    }));
  }

  // Overall is a transparent QC assessment. Model scores keep their individual names.
  const confidenceBreakdown = el('div', { class: 'confidence-breakdown' });
  const confidenceBadge = el('details', { class: 'confidence-details' },
    el('summary', { class: 'confidence-badge', title: 'Show the image quality checks' },
      el('span', { class: 'confidence-dot' }),
      el('span', { class: 'confidence-label' }, 'OVERALL CONFIDENCE'), confidenceValue),
    confidenceBreakdown);
  // The reference-identity key the badge was last built from; see update()'s confidenceKey.
  let lastConfidence = null;

  // The list's status badge (studies-table spec 2026-09-10, section 8.3), so the screen says what the row
  // says: Processing while this study's run is in flight, Reviewed once marked. Rebuilt by update()
  // only when it would change.
  const statusHost = el('div', { class: 'analysis-status' });
  // (2026-10-10, level editor Task 11) The header's eyebrow while editing (handoff §3), before the status badge;
  // update() shows it only in edit mode.
  const editingEyebrow = el('div', { class: 'analysis-meta analysis-editing', hidden: true }, 'EDITING SEGMENTATION');

  // The same DEMO pill the Studies list puts beside the patient. A demo study's numbers are
  // fabricated for exploring the interface; the header is where the user is looking when they
  // read them, so the pill belongs beside the id, not only back on the list.
  // (2026-10-19) Previous / next beside Back, with the open study's place in the Find list between them: what the
  // arrow keys do (handleNavKey below), for the mouse, and where you are. update() keeps the three current.
  const prevButton = el('button', {
    type: 'button', class: 'icon-btn analysis-nav-prev', 'aria-label': 'Previous study', innerHTML: PREV_SVG,
    onClick: () => openNeighbour(-1),
  });
  const nextButton = el('button', {
    type: 'button', class: 'icon-btn analysis-nav-next', 'aria-label': 'Next study', innerHTML: NEXT_SVG,
    onClick: () => openNeighbour(1),
  });
  const position = el('span', { class: 'analysis-meta analysis-position' });
  const navGroup = el('div', { class: 'analysis-nav', role: 'group', 'aria-label': 'Previous and next study' },
    prevButton, position, nextButton);
  // What the group was last built from; update() runs on every notification, pan frames included, and the list's
  // order is a sort over the library, so it is recomputed only when one of its inputs changes.
  let lastNav = null;

  const header = el('header', { class: 'analysis-header' },
    backButton,
    navGroup,
    nameField,
    headerMeta,
    study.source === 'demo' ? el('span', { class: 'pill-demo' }, 'DEMO') : null,
    el('div', { class: 'analysis-spacer' }),
    editingEyebrow,
    statusHost,
    confidenceBadge);

  const tabMeas = el('button', {
    type: 'button', class: 'analysis-tab', onClick: () => setState({ tab: 'meas' }),
  }, 'Measurements');
  const tabSim = el('button', {
    type: 'button', class: 'analysis-tab', onClick: () => setState({ tab: 'sim' }),
  }, 'Find similar');

  const exportButton = el('button', {
    type: 'button', class: 'btn btn-small analysis-export', onClick: () => exportCsv(),
  }, 'Export CSV');

  // Mark reviewed (studies-table spec 2026-09-10, section 8.3): one button that toggles the record's review
  // mark, with the reason it is disabled shown beside it -- Chromium shows no tooltip on a
  // disabled control, and the Parameters bar already renders its reason as a visible note. Marking also
  // writes the label record of a film that has none (2026-10-12, D19; toggleReviewed below).
  const reviewButton = el('button', {
    type: 'button', class: 'btn btn-small analysis-review', 'aria-pressed': 'false',
    onClick: () => toggleReviewed(),
  }, 'Mark reviewed');
  const reviewNote = el('span', { class: 'param-export-note analysis-review-note', hidden: true });

  const measurementsHost = el('div', { class: 'analysis-panel-host' });
  const similarHost = el('div', { class: 'analysis-similar is-hidden' },
    'Find similar arrives in a later build.');
  // (2026-10-10, level editor Task 11) The Vertebral levels panel. In edit mode it replaces the tabs, the actions
  // row and both tab bodies; update() swaps them. Its class is in the stage keyboard's own-Tab-order list
  // (components/viewer.js OWN_TAB_ORDER), so Tab moves focus normally inside it.
  const levelHost = el('div', { class: 'analysis-level-host is-hidden' });

  const tabsRow = el('div', { class: 'analysis-tabs' },
    el('div', { class: 'analysis-tabgroup' }, tabMeas, tabSim));
  const actionsRow = el('div', { class: 'analysis-actions' }, exportButton, reviewButton, reviewNote);
  // (2026-10-15, annotation editor round 2, spec §3.6) The label record's comment, read-only, under the actions row.
  // It shows only while the study has one and hides in edit mode, where the panel's own box takes over.
  const commentText = el('p', { class: 'analysis-comment-text' });
  const commentBlock = el('div', { class: 'analysis-comment is-hidden' },
    el('span', { class: 'analysis-comment-label' }, 'COMMENT'),
    commentText);
  const panel = el('aside', { class: 'analysis-panel' },
    tabsRow,
    actionsRow,
    commentBlock,
    measurementsHost,
    similarHost,
    levelHost);

  const viewerHost = el('div', { class: 'analysis-viewer-host' });
  const body = el('div', { class: 'analysis-body' }, viewerHost, panel);
  // The clinical data drawer is the screen's LAST child, full width below the viewer/panel row
  // (spec 9.5). .analysis-screen is a flex column and .analysis-body is flex:1/min-height:0, so
  // the stage shrinks to make room and the drawer is always in view -- the app shell cannot
  // scroll (.app-shell is height:100vh/overflow:hidden), and it must not: the stage's
  // client-to-image hit-testing assumes a fixed stage. Mounted for demo studies too; the
  // component disables its inputs and its Import button for them.
  const clinicalHost = el('section', { class: 'clinical-data' });
  function changeRegion(patch) {
    const live = getState(), open = currentStudy(live);
    // (2026-10-10, ruling R21) Not while editing: the change wipes the study's levels and drops the open editor
    // without the exit prompt. Both selects are disabled then (update() below); this covers a keyboard activation.
    // (2026-10-20, issue #68) Nor during this film's own turn; a run or batch on OTHER films no longer refuses it.
    if (!open || regionLock(open, live).locked) return;
    // Invalidate any sidecar restoration already awaiting I/O as well as pending edits.
    runsByStudy.set(open.id, (runsByStudy.get(open.id) ?? 0) + 1);
    forgetPrediction(open.id);
    const outgoing = imageCache?.studyId === open.id ? imageCache : null;
    if (outgoing) imageCache = null;
    mounted?.viewer.setImages(null);
    if (outgoing) disposeStudyImages(outgoing.images);
    setState({ editing: false, selection: null, selectedLevel: null, levelEditor: null,
      studies: live.studies.map(s => s.id === open.id ? { ...s, ...patch,
        geometry: null, measurements: null, qc: null, reviewedAt: null, predictionId: null,
        processingError: null, processingErrorAt: null } : s) });
    // (2026-10-10) The film type the levels were opened for is gone; a stashed editor goes too.
    forgetLevelEditor(open.id);
    // (2026-10-11, gate feedback G3) So does the film on the stage, and its window with it.
    forgetFilmWindow(open.id);
    mounted?.viewer.setFilmStatus(null);
    // (2026-10-19) The film comes back for the new region: the backend's preview for cervical and full spine, the
    // decoded original otherwise.
    loadOriginal(open.id);
  }
  const regionSelect = el('select', { 'aria-label': 'Spine region', class: 'workspace-folder-select',
    onChange: e => changeRegion({ region: e.target.value, anteriorSide: null }) },
    el('option', { value: 'auto' }, 'Auto detect'), el('option', { value: 'lumbar' }, 'Lumbar'), el('option', { value: 'cervical' }, 'Cervical · HRNET'),
    el('option', { value: 'full_spine' }, 'Full spine · HRNET'));
  const anteriorSelect = el('select', { 'aria-label': 'Cervical anterior image side', class: 'workspace-folder-select',
    onChange: e => changeRegion({ anteriorSide: validAnteriorSide(e.target.value) ? e.target.value : null }) },
    el('option', { value: '' }, 'Auto detect'), el('option', { value: 'left' }, 'Anterior is image left'),
    el('option', { value: 'right' }, 'Anterior is image right'));
  const anteriorLabel = el('label', {}, 'Orientation ', anteriorSelect);
  const regionNote = el('span', { class: 'meas-note' });
  const previewButton = el('button', { type: 'button', class: 'btn btn-small',
    onClick: () => previewOriginal(getState().openId) }, 'View original');
  // (2026-10-21) The bar carried its own Run segmentation for cervical, full-spine and Auto detect films, while a
  // lumbar film's sat in the UNSEGMENTED strip along the bottom of the stage (2026-10-19). The strip now serves every
  // unsegmented film with its film on the stage, and the full card every other, so the bar's button is gone: one place
  // for the action whatever the region. R17 (no run while editing) holds through the stage's card, hidden while editing.
  const regionBar = el('div', { class: 'analysis-region-bar' }, el('label', {}, 'Region ', regionSelect), anteriorLabel, previewButton, regionNote);
  const root = el('main', { class: 'analysis-screen' }, header, regionBar, body, clinicalHost);

  const viewer = mountViewer(viewerHost);
  const measurementsPanel = mountMeasurements(measurementsHost);
  const levelPanel = mountLevelEditor(levelHost);
  const clinical = mountClinicalData(clinicalHost);

  viewer.setRunHandler(() => {
    const live = getState();
    // `locating` too: a relocate picker is already open for a run that has not started, and a
    // second click must not raise a second native dialog. It is not a run, so it never claims one.
    // `batch` (batch spec 8.3): no single run starts while a batch is up, including in the window
    // between two films where `running` is null.
    // (2026-10-10, ruling R17) `editing`: a run would replace the levels under the open level editor and drop its
    // unsaved edits without the exit prompt, while edit mode hides the RUNNING card and its Cancel. The Re-run
    // tool is disabled while editing; this covers every other way onto this handler.
    if (live.running || live.batch || locating || live.editing) return;
    segmentStudy(live.openId);
  });

  // Re-hand the cached bitmaps to the fresh viewer, so navigating back into an
  // already-segmented study shows its radiograph instead of a black stage.
  if (imageCache && imageCache.studyId === study.id) viewer.setImages(imageCache.images);
  // Only DECIDED here; the restore itself starts after the first paint, because restoreFilm's
  // live() reads `mounted`. A demo study has no film to restore and no sidecar to read.
  // (2026-10-10, ruling R16 as amended) Nor does a study no model run ever produced (isHandPlaced): its geometry
  // came from the level editor, placed on the decoded original, so it is shown from that original (below).
  const hasGeometry = study.source === 'real' && Boolean(study.measurements && study.geometry);
  const needsRestore = !(imageCache && imageCache.studyId === study.id) && hasGeometry && !isHandPlaced(study);
  const handPlaced = hasGeometry && isHandPlaced(study);

  // The mark is a timestamp on the record (spec 8.1); unmarking writes null. Refused for the reasons
  // reviewBlockedReason lists, which update() also shows beside the button, so a keyboard
  // activation of a disabled-looking button cannot slip through. One new-array write; the saver
  // subscribed in renderer/main.js persists it. Every write that changes the numbers clears it
  // again (spec 8.4) -- none of that is here.
  // (2026-10-12, gate feedback G6; plan D19) Marking -- never unmarking -- also writes the film's label record when it
  // has none yet: the zero-change record of the stored state (renderer/level-editor.js writeReviewRecord, which skips a
  // study with a record or with level edits open or stashed, R40). The mark is set first and neither waits for nor
  // depends on the write (R41); the write says how it went in a toast.
  function toggleReviewed() {
    const live = getState();
    const open = currentStudy(live);
    if (!open) return;
    const pending = Boolean(live.measurementDrafts?.[open.id]);
    if (reviewBlockedReason({ study: open, running: live.running, pending })) return;
    const reviewedAt = isReviewed(open) ? null : new Date().toISOString();
    setState((state) => ({
      studies: state.studies.map((study) => (study.id === open.id ? { ...study, reviewedAt } : study)),
    }));
    if (reviewedAt) writeReviewRecord(open.id);
  }

  // What the header badge last showed; update() runs on every notification, pan frames included,
  // and rebuilds the badge only when this changes.
  let lastBadgeKey = null;

  async function exportCsv() {
    const live = getState();
    // toCsv never writes a demo study, so the guard below (the export button is disabled
    // for a demo study) is what keeps a demo's export from producing a header with no rows.
    const csv = toCsv(live.studies.filter((s) => s.id === live.openId));
    const open = currentStudy(live);
    try {
      const savedTo = await saveCsv({ text: csv, suggestedName: `${open ? open.id : 'export'}.csv` });
      // Cancelling the dialog is not an error and must not toast. saveCsv resolves null.
      if (savedTo) showToast(`Exported to ${savedTo}`);
    } catch (error) {
      showToast(`Could not export: ${error.message}`);
    }
  }

  function update() {
    const live = getState();
    const open = currentStudy(live);
    if (!open) return;

    // `open.pt` is the demo-set PATIENT label, not the PT pelvic-tilt measurement
    // (that is open.measurements.PT). Do not "fix" this to a number.
    // The model that produced the numbers on screen, when the result recorded one. Older
    // records carry no provenance and show nothing extra rather than a guessed name.
    const produced = describeModels(open.qc);
    regionSelect.value = requestedRegion(open);
    anteriorSelect.value = open.anteriorSide ?? '';
    anteriorLabel.hidden = ['lumbar', 'cervical'].includes(studyRegion(open));
    anteriorSelect.options[0].textContent = 'Auto detect';
    anteriorSelect.setAttribute('aria-label', requestedRegion(open) === 'auto' ? 'Anterior image side'
      : studyRegion(open) === 'full_spine' ? 'Full spine anterior image side' : 'Cervical anterior image side');
    // (2026-10-20, issue #68) Locked for this film's own turn, edit mode (ruling R21: changing either wipes the study,
    // levels included) and a demo study, each with its tooltip; a run or batch on other films leaves both live.
    const lock = regionLock(open, live, 'region');
    const orientationLock = regionLock(open, live, 'orientation');
    regionSelect.disabled = lock.locked;
    regionSelect.title = lock.title;
    anteriorSelect.disabled = orientationLock.locked;
    anteriorSelect.title = orientationLock.title;
    // View original (the backend's preview, the evidence for the anterior-side choice) for a cervical, full-spine or
    // Auto detect film with no result; Run segmentation is the stage's (the UNSEGMENTED strip or card, 2026-10-21).
    const alignmentSetup = requiresAnteriorSide(open) && !open.measurements;
    previewButton.hidden = !alignmentSetup;
    previewButton.disabled = Boolean(live.running || live.batch);
    // The preview message shows only while the film has no result (2026-10-19: for every real film, since every one
    // now loads its original -- showOriginal -- not only the alignment setup's): it is not cleared when a preview is
    // abandoned, so its being set is not enough. A failed lumbar film keeps its note to the stored reason (port spec 6):
    // the FAILED card covers the stage, and a film the run could not find is not also "unavailable" after it.
    const previewNote = mounted?.previewMessage && open.source === 'real' && !open.measurements
      && (requiresAnteriorSide(open) || !open.processingError) ? mounted.previewMessage : '';
    // (2026-10-01, issue #39) Read here for the region note as well as for the header pill below. A
    // film whose pill reads Failed says why here, in --danger, next to the controls that fix most
    // failures, ahead of the ordinary guidance (port spec 6).
    const { unsupported, status: badgeStatus, badgeKey } = headerBadge(open, live.running, live.batch);
    const failedNote = failedRunNote(open, { unsupported, status: badgeStatus }, previewNote);
    regionNote.classList.toggle('is-failed', failedNote !== null);
    regionNote.textContent = failedNote !== null ? failedNote : previewNote ? previewNote : open.measurements
      ? `${requestedRegion(open) === 'auto' ? `Detected ${studyRegionLabel(open)}. ` : ''}${open.geometry?.anterior_side ? `Anterior: image ${open.geometry.anterior_side}. ` : ''}Changing ${studyRegion(open) === 'cervical' ? 'region' : 'region or orientation'} clears measurements; run again.`
      : requestedRegion(open) === 'auto' ? 'Detect cervical, lumbar or full spine from the film. Choose a region or orientation to override.'
      : studyRegion(open) === 'full_spine' ? 'Use a lateral full-spine film. Orientation is detected automatically; left/right overrides it.'
      : studyRegion(open) === 'cervical' ? 'Use a lateral cervical film. Anterior and posterior are identified automatically.' : '';
    // Never overwrite what the user is in the middle of typing.
    if (document.activeElement !== nameField) {
      nameField.value = studyName(open);
      nameField.size = Math.min(34, Math.max(8, nameField.value.length + 1));
    }
    // Demo records are never written (the saver filters them), so a rename would silently vanish
    // at the next launch. The clinical drawer refuses the same edit for the same reason.
    nameField.disabled = open.source === 'demo';
    nameField.title = open.source === 'demo'
      ? 'Demo studies are not saved'
      : `Rename this study · ${open.id}${open.fileName ? ` · ${open.fileName}` : ''}`;

    // The rest of the header line. The name leads because that is what the user recognises; the
    // SP-nnnn id stays reachable on the title rather than disappearing entirely.
    // Where the models ran, from the providers the result recorded; nothing for older records.
    // (2026-10-19, final review) The region reads as the Find list's REGION cell does (data/cervical.js regionKey and
    // regionLabel), so an Auto detect film no run has placed reads UNKNOWN here too, not AUTO.
    const processedOn = describeProcessor(open.qc);
    headerMeta.textContent = `${regionLabel(regionKey(open)).toUpperCase()} · ${(open.view || 'Unknown').toUpperCase()} · ${open.pt ?? '—'}`
      + (produced ? ` · ${produced.toUpperCase()}` : '') + (processedOn ? ` · ${processedOn}` : '');
    headerMeta.title = processorTitle(open.qc);
    // A run in flight counts as pending too: the numbers on the record are the PREVIOUS run's,
    // so an assessment of them would be a stale claim about a study that is being re-measured.
    const pendingForConfidence = Boolean(live.measurementDrafts?.[open.id]) || live.running === open.id;
    // Cheap gate, on reference identity: update() runs on every notification, pan frames
    // included, and imageConfidence walks the geometry and builds seven detail strings. The
    // store replaces records rather than mutating them, so a changed reference IS a changed
    // input; this replaces the JSON.stringify of the whole assessment that used to run per
    // notification. No setState here -- update() runs inside a store notification.
    const confidenceKey = [open.qc, open.geometry, open.measurements, open.calibration, pendingForConfidence];
    if (!sameConfidenceKey(confidenceKey, lastConfidence)) {
      lastConfidence = confidenceKey;
      const assessment = imageConfidence(open, pendingForConfidence);
      confidenceValue.textContent = assessment.label;
      confidenceBadge.dataset.tone = assessment.tone;
      confidenceBreakdown.replaceChildren(...assessment.details.map(detail => el('p', {}, detail)));
    }

    // toCsv already drops demo rows, so exporting a demo study would write a header and no
    // data. Disabling the button says why instead of handing back an empty file.
    const isDemo = open.source === 'demo';
    const pendingMeasurement = Boolean(live.measurementDrafts?.[open.id]);
    exportButton.disabled = isDemo || pendingMeasurement;
    // No tooltip on the enabled button: it would only repeat the label it sits on.
    exportButton.title = isDemo ? 'Demo studies are not exported'
      : pendingMeasurement ? 'Wait for measurements to finish updating' : '';

    // Mark reviewed (spec 8.3): enabled only when there is something to review and nothing is about
    // to change it. The text carries the date once marked; the note carries the reason otherwise.
    const reason = reviewBlockedReason({ study: open, running: live.running, pending: pendingMeasurement });
    const reviewed = isReviewed(open);
    reviewButton.disabled = reason !== null;
    reviewButton.textContent = reviewed ? reviewedLabel(open.reviewedAt) : 'Mark reviewed';
    reviewButton.setAttribute('aria-pressed', reviewed ? 'true' : 'false');
    reviewButton.title = reviewed && reason === null ? 'Unmark reviewed' : '';
    reviewNote.textContent = reason ?? '';
    reviewNote.hidden = reason === null;

    // (Annotation editor round 2, spec §3.6) The comment block, written every time like the review button: the
    // study's label comment, shown while it has one and edit mode is off. `editing` is a real boolean, here and in the
    // panel swap below: classList.toggle with an undefined force toggles instead of setting.
    const editing = Boolean(live.editing);
    const comment = typeof open.labelComment === 'string' && open.labelComment.trim() !== '' ? open.labelComment : null;
    commentText.textContent = comment ?? '';
    commentBlock.classList.toggle('is-hidden', editing || comment === null);

    // (2026-10-19) Previous / next and the position, from the Find list's order under its filters (data/find.js): every
    // store key that order reads is in the gate, and edit mode disables both buttons (openNeighbour refuses it too).
    const navKey = [live.studies, live.query, live.findSort, live.findFilters, live.running, live.batch, editing];
    if (!sameConfidenceKey(navKey, lastNav)) {
      lastNav = navKey;
      const place = findPosition(live, open.id);
      const before = findNeighbour(live, open.id, -1);
      const after = findNeighbour(live, open.id, 1);
      position.textContent = positionLabel(place);
      position.title = place.index === null
        ? `Not in the Find list under its current filters, which show ${place.count} ${place.count === 1 ? 'study' : 'studies'}`
        : `Study ${place.index} of ${place.count} in the Find list`;
      prevButton.disabled = editing || !before;
      nextButton.disabled = editing || !after;
      prevButton.title = editing ? 'Finish editing before moving to another study'
        : before ? `Previous: ${studyName(before)} (←)` : 'First study in the list';
      nextButton.title = editing ? 'Finish editing before moving to another study'
        : after ? `Next: ${studyName(after)} (→)` : 'Last study in the list';
    }

    // The list's badge, on the list's rule (screens/studies.js buildRow): headerBadge's `unsupported`,
    // `status` and `badgeKey`, read above for the region note. A Failed pill carries the dated reason
    // as its tooltip (2026-10-01, issue #39).
    if (badgeKey !== lastBadgeKey) {
      lastBadgeKey = badgeKey;
      mount(statusHost, unsupported ? unsupportedViewBadge(open.view)
        : statusBadge(badgeStatus, badgeStatus === 'fail' ? failureTitle(open.processingError, open.processingErrorAt) : undefined));
    }

    tabMeas.classList.toggle('is-active', live.tab === 'meas');
    tabSim.classList.toggle('is-active', live.tab === 'sim');
    // (2026-10-10, level editor Task 11) Edit mode swaps the panel: the tabs, the actions row and both tab bodies
    // hide and the Vertebral levels panel shows; DONE brings them back on the tab that was active. `editing` is read
    // above, beside the comment block.
    panel.classList.toggle('is-editing', editing);
    tabsRow.classList.toggle('is-hidden', editing);
    actionsRow.classList.toggle('is-hidden', editing);
    measurementsHost.classList.toggle('is-hidden', editing || live.tab !== 'meas');
    similarHost.classList.toggle('is-hidden', editing || live.tab !== 'sim');
    levelHost.classList.toggle('is-hidden', !editing);
    editingEyebrow.hidden = !editing;

    viewer.updateViewer(open);
    measurementsPanel.updateMeasurements(open);
    // Same contract: every notification, and the panel's own gate decides whether to rebuild.
    levelPanel.update(live, open);
    // Same contract as updateMeasurements: this runs on EVERY store notification, pan frames
    // included, and the component's own reference-keyed gate decides whether to rebuild. It
    // reads the store itself, so it takes no argument.
    clinical.update();
  }

  // (2026-10-19) The arrow keys, on window like the viewer's shortcuts: Left/Up the previous row of the Find list,
  // Right/Down the next, wherever focus is except inside a control that owns its arrows (KEEPS_ARROWS), with no
  // modifier, and never in edit mode, where the viewer's own handler -- registered first, by mountViewer above --
  // nudges handles with them (openNeighbour refuses edit mode). A key with nowhere to go is left alone, so it keeps
  // whatever it did before. teardown() removes the listener with the mount.
  // (2026-10-19, final review; ruling R14) The region bar's Region and Orientation selects do NOT own the arrows. On a
  // closed select Chromium's default for an arrow is "pick the next option", which fires `change`, and changeRegion then
  // wipes the film's result (geometry, measurements, qc, the review mark) with no confirmation: a Right pressed by habit
  // after choosing a region destroyed a segmentation. So with focus on one of them the key is always prevented, which
  // stops the value change, and navigates as from anywhere else; at either end of the list it is still prevented and so
  // changes nothing. The dropdown opened with Space or Alt+Down (Alt returns above) takes its own arrows inside its
  // popup, which Chromium handles before the page sees a keydown, so choosing a region from the keyboard still works.
  // Every other select, input, textarea and contenteditable keeps its arrows.
  function handleNavKey(event) {
    const step = NAV_STEP[event.key];
    if (!step || event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    const target = event.target instanceof Element ? event.target : null;
    if (target && target.closest(REGION_BAR_SELECT)) {
      event.preventDefault();
      openNeighbour(step);
      return;
    }
    if (target && target.closest(KEEPS_ARROWS)) return;
    if (openNeighbour(step)) event.preventDefault();
  }
  window.addEventListener('keydown', handleNavKey);

  // mounted.studyId is refreshed ONLY here, and render() runs only when the router sees a
  // SCREEN_KEYS ('screen', 'calibrationRequest') change or, on this screen, an openId change (router.js
  // ANALYSIS_KEYS, 2026-10-19: previous / next change openId with the screen unchanged). So the two stay in step:
  // every opener remounts this host, and the module-scope subscription above skips an update whose openId is not
  // this mount's. The three readers of studyId -- the setImages guard (~l.203), live() (~l.230) and needsRestore --
  // depend on that, or one study's geometry would be drawn over another study's film.
  mounted = { viewer, update, studyId: study.id, detachKeys: () => window.removeEventListener('keydown', handleNavKey) };
  update();
  // (2026-10-11, ruling R35) A window the view kept for this film (Back, then the sidebar's OPEN STUDY card) over images
  // with no decoded original -- another study's run took the single cache entry meanwhile, so the restore below would
  // hand the viewer the stored image alone -- would read W / L on the chip while the film layer draws the stored image.
  // ensureFilm (through redecodeKeptWindow) restores the stored image, attaches the decode and hands both to this
  // viewer, so the window is drawn again; if no original comes back the window is forgotten.
  // (2026-10-19, final review; ruling R15) The restore and the kept window's redecode start at once: they are a segmented
  // film's own stored image. The three branches after them -- the backend's preview, the hand-placed or level-editor
  // film's decode, and an unsegmented lumbar film's decode -- are chosen here but started only after a short settle
  // (FILM_SETTLE_MS), and only if this mount is still the one on screen: arrow-stepping through a long list of
  // unsegmented films otherwise fired a backend preview or a full decode for every film passed over on the way. The
  // stage shows the UNSEGMENTED (or FAILED) card meanwhile, over the toolbar as during the decode, so the film is on the
  // stage about 150 ms later than before: a click on the Edit tool in that time lands on the card.
  const thisMount = mounted;
  const settled = (load) => setTimeout(() => {
    if (mounted === thisMount && mounted.studyId === study.id) load(study.id);
  }, FILM_SETTLE_MS);
  if (windowNeedsDecode(getState(), study, cachedImages(study.id))) redecodeKeptWindow(study.id);
  else if (needsRestore) restoreFilm(study.id);
  else if (!study.geometry && requiresAnteriorSide(study) && !(imageCache?.studyId === study.id && imageCache.images?.preview)) settled(previewOriginal);
  // (2026-10-10) Back on the screen still editing (the sidebar's OPEN STUDY card keeps the level editor) with
  // no film cached for this study: load it, as entering edit mode did.
  // (2026-10-10, ruling R16) A study whose geometry came from the level editor alone has no sidecar and is shown
  // from its decoded original, the image its levels were placed on.
  else if ((handPlaced || getState().levelEditor?.studyId === study.id) && !(imageCache?.studyId === study.id)) settled(ensureFilm);
  // (2026-10-19) Every other real film with no result shows its X-ray by default (showOriginal): the decoded original
  // at its default window, with the UNSEGMENTED card drawn as a strip along the bottom of the stage once it is there
  // (components/viewer.js describeCard). The cache check is the same as the branches above: a film already decoded was
  // re-handed to the viewer by the setImages call at the top of this section.
  else if (study.source === 'real' && !study.geometry && !(imageCache?.studyId === study.id)) settled(showOriginal);
  return root;
}
