/**
 * Status derivation (spec 13.1, architecture contract "renderer/data/status.js").
 * Status is never stored on a Study — it is computed from measurements, qc and
 * the latest processing error
 * every time it is needed. Pure. The residual threshold is measurements.js's,
 * re-exported, so the panel's consistency warning and the list's status can never
 * disagree. (2026-09-10) The review mark IS stored -- `reviewedAt` -- and the derivation reads
 * it: a marked study is `ok` whatever its qc says unless a later run failed. (2026-10-01, issue #39)
 * A study with no result and no failure is `unseg`; `proc` is displayStatus's alone, for the running
 * study and the films waiting in the running batch
 * (docs/superpowers/specs/2026-10-01-failed-status-port-design.md section 3). (2026-10-15, annotation editor
 * round 2, section 4) A real study whose levels were placed by hand and that no model produced is `hand`:
 * after a failure, no result and the review mark, before the qc reasons.
 */

import { piResidual, RESIDUAL_LIMIT } from './measurements.js';
import { studyRegion } from './cervical.js';
import { isQueued } from './batch.js';
import { isHandPlaced } from './predictions.js';

export { RESIDUAL_LIMIT };
export const CONFIDENCE_LIMIT = 0.6;
// Review threshold, not an estimate of measurement accuracy.
export const S1_CONFIDENCE_LIMIT = 0.6;

export function landmarkReviewReasons(qc) {
  const reasons = [];
  for (const warning of qc?.film_detection?.warnings ?? []) {
    if (typeof warning === 'string' && warning.trim() && !reasons.includes(warning)) reasons.push(warning);
  }
  if (qc?.film_detection?.qc?.requires_review && !reasons.length) {
    reasons.push('Verify the automatically detected film region and image orientation.');
  }
  if (qc?.models?.vertebrae === 'dual_hrnet' || qc?.global_sva) {
    if (qc?.coverage?.partial || qc?.global_sva?.coverage?.partial) reasons.push('Partial full-spine landmarks — only measurements with usable cervical, lumbar and pelvic landmarks are shown.');
    if (qc?.manual_edits?.landmarks) reasons.push('Manually edited landmarks — verify the corrected positions.');
    reasons.push('Verify the C7 centroid, S1 endplate and selected anterior image side.');
    for (const warning of qc?.warnings ?? []) if (typeof warning === 'string' && !reasons.includes(warning)) reasons.push(warning);
    return reasons;
  }
  if (['cervical_slope_hrnet', 'cervical_hrnet', 'cervical_anatomical_hrnet'].includes(qc?.models?.vertebrae) || ['cervical_slope_hrnet', 'cervical_hrnet', 'cervical_anatomical_hrnet'].includes(qc?.models?.cervical)) {
    if (qc?.coverage?.partial) reasons.push('Partial cervical landmarks — only available measurements are shown.');
    if (qc?.manual_edits?.landmarks) reasons.push('Manually edited landmarks — verify the corrected positions.');
    reasons.push('Verify C2/C7 landmarks and the inferred anterior image side.');
    for (const warning of qc?.warnings ?? []) if (typeof warning === 'string' && !reasons.includes(warning)) reasons.push(warning);
    return reasons;
  }
  if (qc?.coverage?.partial) {
    const missing = Array.isArray(qc.coverage.missing) ? qc.coverage.missing.join(', ') : 'landmarks';
    reasons.push(`Partial segmentation — missing ${missing}. Only available landmarks are measured.`);
  }
  if (qc?.coverage?.unoriented?.length) reasons.push('Anterior/posterior orientation unavailable — only middle disc heights can be measured.');
  if (!qc?.manual_edits?.femoral && qc?.femoral?.qc_pass === false) {
    reasons.push(`Femoral measurements unavailable — ${qc.femoral.reason || 'no usable femoral fit'}.`);
  }
  const confidence = qc?.femoral?.confidence;
  if (!qc?.manual_edits?.femoral && typeof confidence === 'number' && confidence < CONFIDENCE_LIMIT) {
    reasons.push('Low femoral fit confidence — check the femoral landmarks.');
  }
  const framing = qc?.framing;
  // Older results without framing metadata remain readable. A framing record
  // must include usable scores; missing/invalid scores cannot silently pass.
  if (framing) {
    const weak = (score) => !Number.isFinite(score) || score < S1_CONFIDENCE_LIMIT || score > 1;
    if (weak(framing.s1_confidence)) reasons.push('S1 detection needs review — check the S1 endplate.');
    if (framing.searched && weak(framing.search_confidence)) {
      reasons.push('Spine location needs review — check the crop and vertebral levels.');
    }
  }
  if (qc?.manual_edits?.landmarks) reasons.push('Manually edited landmarks — verify the corrected positions. Original model scores do not assess these edits.');
  return reasons;
}

export function reviewReasons(study) {
  if (!study?.measurements) return [];
  const reasons = landmarkReviewReasons(study.qc);
  if (studyRegion(study) === 'full_spine' && !reasons.includes('Verify the C7 centroid, S1 endplate and selected anterior image side.')) {
    reasons.push('Verify the C7 centroid, S1 endplate and selected anterior image side.');
  }
  if (piResidual(study.measurements) > RESIDUAL_LIMIT) {
    reasons.unshift('Parameters inconsistent — check S1 and femoral landmarks.');
  }
  return reasons;
}

// First match wins (port spec 3). A failure outranks measurements and the review mark (1.0.13's
// rule); no result and no failure is Unsegmented, never Processing. (2026-10-15, annotation editor round 2,
// section 4.1) The order is fail, unseg, ok, hand, rev, seg: a marked hand-placed film reads Reviewed, and a real
// study with no model run reads Hand-placed. The real-study guard is load-bearing: a demo has neither model
// field (no prediction id, no qc.models) and must keep reading Segmented.
/** @returns {'fail'|'unseg'|'ok'|'hand'|'rev'|'seg'} */
export function deriveStatus(study) {
  if (!study) return 'unseg';
  if (study.processingError) return 'fail';
  if (study.measurements == null) return 'unseg';
  if (isReviewed(study)) return 'ok';
  if (study.source === 'real' && isHandPlaced(study)) return 'hand';
  return reviewReasons(study).length ? 'rev' : 'seg';
}

// The review mark (studies-table spec 2026-09-10, section 8.1): a non-blank string on the record. It is an
// ISO timestamp -- validateStudy drops anything that is not a date -- but the status asks only
// whether a person set it. It outranks every qc reason (spec 8.2), but not a failed rerun; the reasons themselves stay, and
// the Measurements panel keeps showing them after the review.
export function isReviewed(study) {
  return typeof study?.reviewedAt === 'string' && study.reviewedAt.trim() !== '';
}

// The status a row or a header SHOWS: spec 13.1's "or currently running" half, which is a property
// of state.running and not of the record. deriveStatus stays a pure function of the record; every
// surface that badges a study calls this with state.running, so the list, the summary, the sort
// and the Analysis header cannot disagree. (2026-10-01, issue #39; port spec 3) `batch` is
// state.batch: every film still waiting in a running batch reads Processing too, from the click that
// starts it until its own turn ends; once the batch is stopping, the films that will not run read
// their own status again. This is the only source of 'proc'.
/** @returns {'proc'|'fail'|'unseg'|'ok'|'hand'|'rev'|'seg'} */
export function displayStatus(study, runningId = null, batch = null) {
  if (study && runningId !== null && runningId === study.id) return 'proc';
  if (study && batch && !batch.stopping && isQueued(batch, study.id)) return 'proc';
  return deriveStatus(study);
}

// (2026-10-01, issue #39) Every key is named; anything else is an absent value, the em dash. An
// unrecognised key must never claim a run.
export function statusLabel(status) {
  if (status === 'proc') return 'Processing';
  if (status === 'unseg') return 'Unsegmented';
  if (status === 'fail') return 'Failed';
  if (status === 'seg') return 'Segmented';
  if (status === 'rev') return 'Needs review';
  if (status === 'hand') return 'Hand-placed';
  if (status === 'ok') return 'Reviewed';
  return '\u2014';
}

const reviewedDate = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', year: 'numeric' });

// The Analysis screen's button once marked (spec 8.3): `Reviewed \u00B7 Sep 10, 2026`, or just `Reviewed`
// when the stored value does not parse. It cannot fail to parse after validateStudy, but the label
// never guesses a date.
export function reviewedLabel(reviewedAt) {
  const time = Date.parse(reviewedAt ?? '');
  return Number.isNaN(time) ? 'Reviewed' : `Reviewed \u00B7 ${reviewedDate.format(new Date(time))}`;
}

export const REVIEW_DEMO = 'Demo studies are not saved';
export const REVIEW_NOTHING = 'Nothing to review yet';
export const REVIEW_RUNNING = 'Wait for the segmentation to finish';
export const REVIEW_PENDING = 'Wait for measurements to finish updating';
export const REVIEW_FAILED = 'Rerun segmentation after the failed attempt';

// Why the Mark reviewed button is disabled, or null when it is enabled (spec 8.3). In the order the
// screen would otherwise contradict itself: a demo is never saved whatever else is true; a study
// whose run is in flight has no numbers to review yet even when old ones are still on the record;
// no measurements means nothing to review; a pending correction means the numbers are about to
// change. `running` is state.running; `pending` is whether a measurement draft exists for it.
export function reviewBlockedReason({ study, running = null, pending = false }) {
  if (!study) return REVIEW_NOTHING;
  if (study.source === 'demo') return REVIEW_DEMO;
  if (running !== null && running === study.id) return REVIEW_RUNNING;
  if (study.processingError) return REVIEW_FAILED;
  if (study.measurements == null) return REVIEW_NOTHING;
  if (pending) return REVIEW_PENDING;
  return null;
}
