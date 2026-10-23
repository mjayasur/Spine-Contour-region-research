import { measurementGeometry } from '../data/measurement-sources.js';
import { segmentalColumns, segmentalEndplates } from '../data/segmental.js';
import { CERVICAL_LEVELS } from '../data/cervical.js';
import { LEVELS, FEMORAL_SIDES } from './geometry.js';
import { applyWindow } from './window-level.js';
import { LEVEL_CODES, kindOf, hasCentroid, levelState, placementSteps, fitRim } from '../data/levels.js';
import { quadrilateral } from '../data/level-projection.js';
import { levelHandles, handleRadiusImagePx, handleScale } from './level-handles.js';
import { labelPlacement, plateOffset } from './level-labels.js';

export const LEVEL_RGB = {
  C2: [183, 139, 244],
  C3: [255, 99, 132],
  C4: [255, 159, 64],
  C5: [255, 205, 86],
  C6: [75, 192, 192],
  C7: [54, 162, 235],
  L1: [255, 99, 132],
  L2: [255, 159, 64],
  L3: [255, 205, 86],
  L4: [75, 192, 192],
  L5: [54, 162, 235],
  // (2026-10-10, level editor, D9) The three levels the editor adds that the mask overlay never painted.
  T1: [255, 159, 64],
  T4: [255, 205, 86],
  S1: [255, 99, 132],
};
export const FEMORAL_OVERLAY_COLOR = [98, 210, 111];
// Baked into the overlay pixel data once per prediction. The static layer scales this by
// (overlayOpacity / 100) via ctx.globalAlpha at draw time, so the default overlayOpacity of 50
// reproduces the legacy renderer.js's hardcoded alpha of 58 exactly (116 * 0.5 = 58).
// That file was deleted by this plan's Task 11; the reference is historical.
export const BASE_OVERLAY_ALPHA = 116;

// The levels the backend's mask paints. LEVEL_RGB also holds T1, T4 and S1 for the level editor's
// outlines; the stored image has never coloured S1 and has no T1 or T4, and still does not.
export const MASK_LEVELS = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'L1', 'L2', 'L3', 'L4', 'L5'];

// (2026-10-22, round 2, Task 17: the owner's report) `levels`, an array of codes, restricts the map to those levels
// (the ones the study's geometry still holds); null colours every level, as before.
export function buildLabelColorMap(labels, levels = null) {
  const map = {};
  for (const level of MASK_LEVELS) {
    if (levels && !levels.includes(level)) continue;
    const id = labels?.[level];
    if (typeof id === 'number') map[id] = LEVEL_RGB[level];
  }
  return map;
}

export function buildOverlayPixels(maskPixels, _femoralPixels, colorByLabel, alpha) {
  const overlay = new Uint8ClampedArray(maskPixels.length);
  for (let offset = 0; offset < maskPixels.length; offset += 4) {
    const labelId = maskPixels[offset];
    const color = colorByLabel[labelId];
    if (color) {
      overlay[offset] = color[0];
      overlay[offset + 1] = color[1];
      overlay[offset + 2] = color[2];
      overlay[offset + 3] = alpha;
    }
  }
  return overlay;
}

export async function bitmapFromBase64(base64) {
  const binary = atob(base64);
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  return createImageBitmap(new Blob([bytes], { type: 'image/png' }));
}

export async function loadStudyImages(predictResponse) {
  const [image, mask] = await Promise.all([
    bitmapFromBase64(predictResponse.image_png),
    bitmapFromBase64(predictResponse.mask_png),
  ]);
  const scratch = document.createElement('canvas');
  scratch.width = image.width;
  scratch.height = image.height;
  const context = scratch.getContext('2d');
  context.drawImage(mask, 0, 0);
  const maskPixels = context.getImageData(0, 0, image.width, image.height).data;
  const colorByLabel = buildLabelColorMap(predictResponse.labels);
  const overlayPixels = buildOverlayPixels(maskPixels, null, colorByLabel, BASE_OVERLAY_ALPHA);
  context.clearRect(0, 0, image.width, image.height);
  context.putImageData(new ImageData(overlayPixels, image.width, image.height), 0, 0);
  // `labels` ({ L1: 20, ... }) rides with `mask` so the label record can trace a level's outline from it later
  // (Task 12). The RGBA buffer read above is not kept: `mask` is the source and a film's buffer is 4 bytes a pixel.
  return { image, mask, overlayCanvas: scratch, width: image.width, height: image.height, labels: predictResponse.labels ?? null };
}

// The overlay canvas for a given set of level codes, rebuilt from the stored mask (null = every level). A new
// canvas each call: the images object the screen caches is never changed in place.
export function overlayCanvasFor(images, levels) {
  if (!images?.mask) return images?.overlayCanvas ?? null;
  const scratch = document.createElement('canvas');
  scratch.width = images.width;
  scratch.height = images.height;
  const context = scratch.getContext('2d');
  context.drawImage(images.mask, 0, 0);
  const maskPixels = context.getImageData(0, 0, images.width, images.height).data;
  const overlayPixels = buildOverlayPixels(maskPixels, null, buildLabelColorMap(images.labels, levels), BASE_OVERLAY_ALPHA);
  context.clearRect(0, 0, images.width, images.height);
  context.putImageData(new ImageData(overlayPixels, images.width, images.height), 0, 0);
  return scratch;
}

// `image` is an ImageBitmap, except for a film shown from its decoded original (imagesFromRaw),
// whose `image` is a canvas: nothing to close.
export function disposeStudyImages(images) {
  if (!images) return;
  images.image.close?.();
  images.mask?.close();
}

// (2026-10-10, level editor) The windowed film's RGBA, reused while the film size is unchanged, so a
// slider tick or a drag frame on a 25-megapixel film does not allocate 100 MB.
let windowScratch = null;

function windowedPixels(raw, window) {
  const size = raw.width * raw.height * 4;
  if (!windowScratch || windowScratch.length !== size) windowScratch = new Uint8ClampedArray(size);
  return applyWindow(raw, window, windowScratch);
}

// A film with no stored image (no segmentation result, or none yet) shown from its decoded original
// at `window`: the same shape loadStudyImages returns, with no mask and no overlay, `raw` attached
// and `preview: true` -- it is never a prediction. A canvas is a valid drawImage source.
export function imagesFromRaw(raw, window) {
  const image = document.createElement('canvas');
  image.width = raw.width;
  image.height = raw.height;
  image.getContext('2d').putImageData(new ImageData(windowedPixels(raw, window), raw.width, raw.height), 0, 0);
  return { image, mask: null, overlayCanvas: null, width: raw.width, height: raw.height, raw, preview: true };
}

// Spec 13: a thumbnail is at most 128 px on its long edge, JPEG, inline as a data URI. Pure size
// math here so it can be tested; the encode below needs a canvas.
export function thumbnailSize(width, height, maxEdge = 128) {
  const longEdge = Math.max(width, height, 1);
  const scale = Math.min(1, maxEdge / longEdge);
  return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))];
}

export function thumbnailDataUri(bitmap, maxEdge = 128) {
  const [width, height] = thumbnailSize(bitmap.width, bitmap.height, maxEdge);
  const scratch = document.createElement('canvas');
  scratch.width = width;
  scratch.height = height;
  scratch.getContext('2d').drawImage(bitmap, 0, 0, width, height);
  return scratch.toDataURL('image/jpeg', 0.82);
}

export function createLayeredCanvases(host) {
  const staticCanvas = document.createElement('canvas');
  const dynamicCanvas = document.createElement('canvas');
  // Layout lives in styles/screens/analysis.css under .viewer-canvas. These used to be
  // five inline style writes pinning both canvases to width:100%/height:100%, which
  // stretched every radiograph to the stage's aspect ratio.
  staticCanvas.className = 'viewer-canvas';
  dynamicCanvas.className = 'viewer-canvas viewer-canvas-dynamic';
  host.append(staticCanvas, dynamicCanvas);
  return {
    staticCanvas,
    dynamicCanvas,
    staticCtx: staticCanvas.getContext('2d'),
    dynamicCtx: dynamicCanvas.getContext('2d'),
  };
}

export function sizeCanvases(canvases, width, height) {
  canvases.staticCanvas.width = width;
  canvases.staticCanvas.height = height;
  canvases.dynamicCanvas.width = width;
  canvases.dynamicCanvas.height = height;
}

// The film, then the mask overlay. (2026-10-10, level editor) `opts.window` ({width, level}) with a
// decoded original on `images.raw` puts the windowed original instead of the stored image -- the
// LUT touches the film layer only -- and `opts.editing` leaves the overlay off. With neither set,
// the calls are exactly what they were.
export function drawStaticLayer(ctx, canvas, images, opts) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!images) return;
  if (opts.window && images.raw) {
    const { raw } = images;
    ctx.putImageData(new ImageData(windowedPixels(raw, opts.window), raw.width, raw.height), 0, 0);
  } else {
    ctx.drawImage(images.image, 0, 0);
  }
  const overlayCanvas = opts.overlayCanvas ?? images.overlayCanvas;
  if (opts.overlays && overlayCanvas && !opts.editing) {
    ctx.globalAlpha = Math.max(0, Math.min(1, opts.overlayOpacity / 100));
    ctx.drawImage(overlayCanvas, 0, 0);
    ctx.globalAlpha = 1;
  }
}

// The four off-theme literals the architecture contract sanctions for the stage itself;
// plan 04's edit-mode literals follow CANVAS_MONO below. Everything drawn as DOM
// over this canvas is styled from styles/screens/analysis.css -- see BD-3.
const STAGE_LINE_COLOR = '#38342F';
const STAGE_SELECTED_COLOR = '#D45A32';
const STAGE_LABEL_FILL = 'rgba(250,247,242,.75)';
const LABEL_PLATE_FILL = 'rgba(11,10,9,.78)';

// Canvas text cannot express font-variant-numeric: tabular-nums (the ctx.font shorthand
// has no slot for it), so every canvas-drawn label that contains a number uses Chivo
// Mono, which is monospaced and therefore tabular by construction.
const CANVAS_MONO = "'Chivo Mono', monospace";

// Plan 04 additions to this file's literal set. All of them are pixels drawn INTO the
// canvas, which is the exception the architecture contract grants viewer/canvas.js:
// the stage background (the contract's first literal) as the handle outline; the legacy
// editor's four per-corner colours (renderer.js:43, historical); the femoral handle colour,
// which is the overlay's own femoral green; and the retrace point colour.
const STAGE_BG_COLOR = '#0B0A09';
export const CORNER_COLORS = { SA: '#32d4ff', SP: '#64e19a', IA: '#ffb259', IP: '#fa78d4' };
const FEMORAL_HANDLE_COLOR = `rgb(${FEMORAL_OVERLAY_COLOR.join(',')})`;
export const TRACE_COLOR = '#ffe071';

// Handles keep a constant size ON SCREEN up to 1:1 (past it they magnify with the film, R28), unlike the
// outlines, which scale with the image; the level handle itself shrinks with the film below 1:1 (G8, D21).
// Every handle dimension below is in CSS pixels and is multiplied by opts.pixelRatio
// (image pixels per CSS pixel at the current fit and zoom, floored at 1 by ruling R28) at draw time.
const HANDLE_RING_RADIUS_PX = 10;

function midpoint(a, b) {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
}

// Draws the level's name (L1..L5, S1) ONLY when that level is selected. Named for what
// it does: the unselected levels are identified by their outline, not by a label, so the
// stage is not covered in text. Sized with the image, like the outlines.
function drawSelectedStageLabel(ctx, text, point, selected, canvasWidth) {
  if (!selected) return;
  const fontSize = Math.max(8, canvasWidth / 90);
  ctx.font = `700 ${fontSize}px ${CANVAS_MONO}`;
  ctx.fillStyle = STAGE_LABEL_FILL;
  ctx.fillText(text, point[0] + fontSize * 0.5, point[1] - fontSize * 0.5);
}

// The point just beyond an endplate's anterior corner (15% of the endplate's length past
// it, along the endplate) and which way a label should extend from it, away from the
// body: +1 rightward, -1 leftward. Scale-free, so it holds at any film size.
function beyondAnterior(sa, sp) {
  const dx = sa[0] - sp[0];
  const dy = sa[1] - sp[1];
  const gap = 0.15;
  return { anchor: [sa[0] + dx * gap, sa[1] + dy * gap], side: dx >= 0 ? 1 : -1 };
}

// The selected construction's label: what it says and where it anchors, in image space.
// The label itself is a DOM chip over the stage (components/viewer.js) so it can sit in the
// black space around the film and scale with it; no measurement text is drawn on the canvas.
// Every selectedLevel value is handled explicitly; anything else has no label.
export function constructionLabel(geometry, selectedLevel, measurements) {
  geometry = measurementGeometry(geometry, selectedLevel === 'GLOBAL_SVA' ? 'global_sva' : selectedLevel === 'PT' ? 'pelvic_tilt' : null);
  if (!geometry || !selectedLevel || !measurements) return null;
  if (selectedLevel.startsWith('SEG_')) {
    const plates = segmentalEndplates(geometry, selectedLevel);
    const value = measurements[selectedLevel];
    if (!plates || !Number.isFinite(value)) return null;
    const def = segmentalColumns('full_spine').find(c => c.key === selectedLevel);
    return { text: `${def.label} ${value.toFixed(1)}°`, ...beyondAnterior(...plates[0]) };
  }
  if (geometry.region === 'full_spine' && selectedLevel === 'GLOBAL_SVA') {
    if (!Number.isFinite(measurements.GLOBAL_SVA_PX)
        || !geometry.c7_centroid || !geometry.s1_superior?.[1]) return null;
    const mm = Number.isFinite(measurements.GLOBAL_SVA_MM);
    return { text: `C7–S1 SVA ${(mm ? measurements.GLOBAL_SVA_MM : measurements.GLOBAL_SVA_PX).toFixed(1)} ${mm ? 'mm' : 'px (uncalibrated)'}`,
      anchor: [geometry.c7_centroid[0], geometry.s1_superior[1][1]], side: 1 };
  }
  if (['cervical', 'full_spine'].includes(geometry.region) && ['C2C7_COBB', 'C2C7_SVA'].includes(selectedLevel)) {
    if (selectedLevel === 'C2C7_COBB' && Number.isFinite(measurements.C2C7_COBB)) {
      return { text: `C2–C7 Cobb ${measurements.C2C7_COBB.toFixed(1)}°`,
        ...beyondAnterior(...geometry.vertebrae.C2.inferior) };
    }
    if (selectedLevel === 'C2C7_SVA' && Number.isFinite(measurements.C2C7_SVA_PX)) {
      const mm = Number.isFinite(measurements.C2C7_SVA_MM);
      return { text: `C2–C7 SVA ${(mm ? measurements.C2C7_SVA_MM : measurements.C2C7_SVA_PX).toFixed(1)} ${mm ? 'mm' : 'px (uncalibrated)'}`,
        anchor: [geometry.c2_centroid[0], geometry.vertebrae.C7.superior[1][1]], side: 1 };
    }
    return null;
  }
  // (2026-10-22, spec P11) The T1 slope reads geometry.t1_superior only, so it labels on a cervical film, which has no
  // S1: this branch sits ABOVE the S1 gate.
  if (selectedLevel === 'T1S') {
    const t1 = geometry.t1_superior;
    if (!isPair(t1) || !Number.isFinite(measurements.T1S)) return null;
    return { text: `T1 slope ${measurements.T1S.toFixed(1)}°`, ...beyondAnterior(t1[0], t1[1]) };
  }
  const s1 = geometry.s1_superior;
  if (!s1) return null;
  const s1Mid = midpoint(s1[0], s1[1]);
  const hip = geometry.hip_midpoint;
  if (selectedLevel === 'S1') {
    const { PI, PT, SS } = measurements;
    if (!hip || PI == null || PT == null || SS == null) return null;
    return { text: `PI ${PI.toFixed(1)}°  PT ${PT.toFixed(1)}°  SS ${SS.toFixed(1)}°`, anchor: midpoint(s1Mid, hip), side: 1 };
  }
  if (selectedLevel === 'SS') {
    return measurements.SS == null ? null : { text: `SS ${measurements.SS.toFixed(1)}°`, ...beyondAnterior(s1[0], s1[1]) };
  }
  if (selectedLevel === 'PT') {
    return measurements.PT == null ? null : { text: `PT ${measurements.PT.toFixed(1)}°`, anchor: midpoint(hip, s1Mid), side: 1 };
  }
  if (selectedLevel === 'PI') {
    return measurements.PI == null ? null : { text: `PI ${measurements.PI.toFixed(1)}°`, anchor: midpoint(s1Mid, hip), side: 1 };
  }
  if (selectedLevel === 'L1PA') {
    return measurements.L1PA == null ? null : { text: `L1PA ${measurements.L1PA.toFixed(1)}°`, anchor: midpoint(hip, geometry.l1_center), side: 1 };
  }
  // (2026-10-22, spec P11) T4 pelvic angle: the angle at the hip between the T4 centroid and the S1 midpoint.
  if (selectedLevel === 'T4PA') {
    const t4 = geometry.t4_centroid;
    if (!isPoint(t4) || !isPoint(hip) || !Number.isFinite(measurements.T4PA)) return null;
    return { text: `T4PA ${measurements.T4PA.toFixed(1)}°`, anchor: midpoint(hip, t4), side: 1 };
  }
  if (LEVELS.includes(selectedLevel)) {
    const body = geometry.vertebrae?.[selectedLevel];
    const key = `${selectedLevel}-S1`;
    const value = measurements.LL?.[key];
    if (!body?.superior || value == null) return null; // (2026-10-22, the final review, M2) a body with no superior pair
    return { text: `LL ${key} ${value.toFixed(1)}°`, ...beyondAnterior(body.superior[0], body.superior[1]) };
  }
  return null;
}

// Reference axes (horizontal, vertical, endplate normal) are drawn dashed and slightly
// faded so it is never ambiguous which line is the anatomy being measured and which is the
// datum it is measured against.
function strokeReference(ctx, from, to) {
  ctx.save();
  ctx.setLineDash([6, 5]);
  ctx.globalAlpha = 0.7;
  ctx.beginPath();
  ctx.moveTo(...from);
  ctx.lineTo(...to);
  ctx.stroke();
  ctx.restore();
}

function drawSelectedMeasurement(ctx, canvas, geometry, selectedLevel, measurements) {
  geometry = measurementGeometry(geometry, selectedLevel === 'GLOBAL_SVA' ? 'global_sva' : selectedLevel === 'PT' ? 'pelvic_tilt' : null);
  if (!selectedLevel || !measurements) return;
  // The same dependencies gate the caption and its construction.
  if (!constructionLabel(geometry, selectedLevel, measurements)) return;
  ctx.save();
  try {
    ctx.strokeStyle = STAGE_SELECTED_COLOR;
    ctx.lineWidth = Math.max(2, canvas.width / 400);
    if (selectedLevel.startsWith('SEG_')) {
      for (const [a, p] of segmentalEndplates(geometry, selectedLevel)) {
        const dx = p[0] - a[0], dy = p[1] - a[1];
        ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...p); ctx.stroke();
        strokeReference(ctx, [a[0] - dx, a[1] - dy], [p[0] + dx, p[1] + dy]);
      }
    } else if (geometry.region === 'full_spine' && selectedLevel === 'GLOBAL_SVA') {
      const c = geometry.c7_centroid, p = geometry.s1_superior[1];
      const foot = [c[0], p[1]];
      strokeReference(ctx, c, foot);
      ctx.beginPath(); ctx.moveTo(...foot); ctx.lineTo(...p); ctx.stroke();
      const tick = Math.max(4, canvas.width / 150);
      for (const x of [foot[0], p[0]]) { ctx.beginPath(); ctx.moveTo(x, p[1] - tick); ctx.lineTo(x, p[1] + tick); ctx.stroke(); }
    } else if (['cervical', 'full_spine'].includes(geometry.region) && ['C2C7_COBB', 'C2C7_SVA'].includes(selectedLevel)) {
      if (selectedLevel === 'C2C7_COBB') {
        for (const level of ['C2', 'C7']) {
          const [a, p] = geometry.vertebrae[level].inferior;
          const dx = p[0] - a[0], dy = p[1] - a[1];
          ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...p); ctx.stroke();
          strokeReference(ctx, [a[0] - dx * 1.5, a[1] - dy * 1.5], [p[0] + dx * 1.5, p[1] + dy * 1.5]);
        }
      } else if (selectedLevel === 'C2C7_SVA') {
        const c = geometry.c2_centroid, p = geometry.vertebrae.C7.superior[1];
        const foot = [c[0], p[1]];
        strokeReference(ctx, c, foot);
        ctx.beginPath(); ctx.moveTo(...foot); ctx.lineTo(...p); ctx.stroke();
        const tick = Math.max(4, canvas.width / 150);
        for (const x of [foot[0], p[0]]) { ctx.beginPath(); ctx.moveTo(x, p[1] - tick); ctx.lineTo(x, p[1] + tick); ctx.stroke(); }
      }
    } else if (selectedLevel === 'T1S') {
      // (2026-10-22, spec P11) T1 slope: the T1 superior endplate against the HORIZONTAL, as SS is drawn. Reads
      // t1_superior only (no S1 on a cervical film); the reference span is the S1-hip distance when both exist.
      const t1 = geometry.t1_superior;
      const t1Mid = midpoint(t1[0], t1[1]);
      const s1 = geometry.s1_superior;
      const hip = geometry.hip_midpoint;
      const span = (isPair(s1) && isPoint(hip) ? Math.hypot(hip[0] - (s1[0][0] + s1[1][0]) / 2, hip[1] - (s1[0][1] + s1[1][1]) / 2) : 0)
        || canvas.width / 6;
      ctx.beginPath();
      ctx.moveTo(...t1[0]);
      ctx.lineTo(...t1[1]);
      ctx.stroke();
      const dir = t1[1][0] >= t1[0][0] ? -1 : 1;
      strokeReference(ctx, t1Mid, [t1Mid[0] + dir * span, t1Mid[1]]);
    } else if (selectedLevel === 'S1') {
      const s1 = geometry.s1_superior;
      const s1Mid = midpoint(s1[0], s1[1]);
      const hip = geometry.hip_midpoint;
      ctx.beginPath();
      ctx.moveTo(...s1Mid);
      ctx.lineTo(...hip);
      ctx.stroke();
    } else if (selectedLevel === 'PI' || selectedLevel === 'PT' || selectedLevel === 'SS') {
      const s1 = geometry.s1_superior;
      const s1Mid = midpoint(s1[0], s1[1]);
      const hip = geometry.hip_midpoint;
      // Reference rays are drawn the same length as the S1-to-hip span so the angle reads at a
      // sensible scale on any image size.
      const span = (hip ? Math.hypot(hip[0] - s1Mid[0], hip[1] - s1Mid[1]) : 0) || canvas.width / 6;

      if (selectedLevel === 'SS') {
        // Sacral slope: the S1 superior endplate against the HORIZONTAL.
        // backend: atan2(s1_vector.y, s1_vector.x)
        ctx.beginPath();
        ctx.moveTo(...s1[0]);
        ctx.lineTo(...s1[1]);
        ctx.stroke();
        // The horizontal extends OPPOSITE the endplate's own x-direction. Flipped from the
        // original on clinical review -- it reads better against the sacrum this way. The
        // reported angle is unaffected: the endplate is drawn as a full segment through
        // s1Mid, so the horizontal ray still forms the acute angle with one half of it
        // whichever side it sticks out.
        const dir = s1[1][0] >= s1[0][0] ? -1 : 1;
        strokeReference(ctx, s1Mid, [s1Mid[0] + dir * span, s1Mid[1]]);
      } else if (selectedLevel === 'PT') {
        // Pelvic tilt: the hip-to-S1-midpoint line against the VERTICAL through the hip.
        // backend: atan2(s1_mid.x - hip.x, hip.y - s1_mid.y)
        ctx.beginPath();
        ctx.moveTo(...hip);
        ctx.lineTo(...s1Mid);
        ctx.stroke();
        strokeReference(ctx, hip, [hip[0], hip[1] - span]);
      } else {
        // Pelvic incidence: the S1-midpoint-to-hip line against the PERPENDICULAR to the S1
        // endplate at its midpoint. backend: angle between the connection and normal_angle,
        // where normal_angle = s1_angle - pi/2, i.e. the endplate vector rotated -90deg.
        ctx.beginPath();
        ctx.moveTo(...s1Mid);
        ctx.lineTo(...hip);
        ctx.stroke();
        const vx = s1[1][0] - s1[0][0];
        const vy = s1[1][1] - s1[0][1];
        const length = Math.hypot(vx, vy) || 1;
        let normal = [vy / length, -vx / length];
        // Draw the perpendicular on the side the hip is on, so the two rays open into the angle
        // actually being reported rather than its supplement.
        if (normal[0] * (hip[0] - s1Mid[0]) + normal[1] * (hip[1] - s1Mid[1]) < 0) {
          normal = [-normal[0], -normal[1]];
        }
        strokeReference(ctx, s1Mid, [s1Mid[0] + normal[0] * span, s1Mid[1] + normal[1] * span]);
      }
    } else if (selectedLevel === 'L1PA') {
      // L1 pelvic angle: the angle subtended at the hip midpoint between the L1 body
      // centroid and the S1 endplate midpoint. Two rays from the hip -- NOT an endplate
      // pair. This branch exists because falling through to the lordosis branch below drew
      // the wrong construction under the right label. See the contract's selectedLevel
      // section.
      const hip = geometry.hip_midpoint;
      const l1c = geometry.l1_center;
      const s1Mid = midpoint(geometry.s1_superior[0], geometry.s1_superior[1]);
      ctx.beginPath();
      ctx.moveTo(...hip);
      ctx.lineTo(...l1c);
      ctx.moveTo(...hip);
      ctx.lineTo(...s1Mid);
      ctx.stroke();
    } else if (selectedLevel === 'T4PA') {
      // (2026-10-22, spec P11) T4 pelvic angle: two rays from the hip, to the T4 centroid and to the S1 midpoint,
      // as L1PA is drawn, never the lordosis branch below.
      const hip = geometry.hip_midpoint;
      const t4 = geometry.t4_centroid;
      const s1Mid = midpoint(geometry.s1_superior[0], geometry.s1_superior[1]);
      ctx.beginPath();
      ctx.moveTo(...hip);
      ctx.lineTo(...t4);
      ctx.moveTo(...hip);
      ctx.lineTo(...s1Mid);
      ctx.stroke();
    } else if (LEVELS.includes(selectedLevel)) {
      // Explicit branch, not a catch-all `else`. The architecture contract's selectedLevel
      // section makes this a binding rule: anything switching on selectedLevel must handle
      // non-level values explicitly rather than falling into a branch that assumes a
      // vertebra -- that exact shape is how L1PA once drew the lordosis line under its own
      // label. selectedLevel's domain also includes 'S1' | 'PI' | 'PT' | 'SS' | 'L1PA' | null,
      // all handled above; this branch is reached only for an actual vertebral level.
      const body = geometry.vertebrae?.[selectedLevel];
      const s1 = geometry.s1_superior;
      if (body?.superior) { // (2026-10-22, the final review, M2) no superior pair, no LL line
        ctx.beginPath();
        ctx.moveTo(...body.superior[0]);
        ctx.lineTo(...body.superior[1]);
        ctx.moveTo(...s1[0]);
        ctx.lineTo(...s1[1]);
        ctx.stroke();
      }
    }
  } finally {
    ctx.restore();
  }
}

// ---- Level editor (2026-10-10) ----
//
// What the level editor draws on the dynamic layer while editing (handoff §5; plan Task 9): every visible
// placed level filled and outlined, the selected level's handles, the model's own geometry dashed under an
// edited level, a plate naming each level, and the placement in progress. Every size is in CSS pixels times
// `opts.pixelRatio` floored at 1 (ruling R28), so it holds its size on screen at any fit and zoom up to 1:1 and
// magnifies with the film past it; the one exception is the level handle (the disc, its outline and its focus ring,
// and the placement points), sized against the UNFLOORED ratio: 6 CSS px at 1:1, growing past it with the square
// root of the zoom rather than with the film (gate feedback G9, decision D22 as amended), and shrinking with the film
// below 1:1 to a floor of 3 CSS px, outline and ring in step (gate feedback G8, decision D21; `handleRadiusImagePx`,
// `handleScale`). The editor state is read, never written.

const LEVEL_STROKE_ALPHA = 0.85;
const LEVEL_STROKE_PX = 1.25;
const SELECTED_STROKE_PX = 2;
const GHOST_COLOR = 'rgba(250,247,242,.7)';
const INK_COLOR = '#FAF7F2';
const LEVEL_HANDLE_OUTLINE_PX = 2;
const CENTRE_DOT_RADIUS_PX = 3.5;
const C2_CENTROID_RADIUS_PX = 3;
const T4_CENTROID_RADIUS_PX = 5;
const PLACEMENT_LINE_PX = 1.5;
const EDITOR_DASH_PX = [6, 5];
const LABEL_FONT_PX = 11;
const LABEL_HEIGHT_PX = 18;
const LABEL_PAD_PX = 6;
const LABEL_RADIUS_PX = 4;
const LABEL_OFFSET_PX = 14;

const isPoint = (p) => Array.isArray(p) && Number.isFinite(p[0]) && Number.isFinite(p[1]);
const isPair = (pair) => Array.isArray(pair) && pair.length === 2 && isPoint(pair[0]) && isPoint(pair[1]);
const rgba = (color, alpha) => `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;
const dashFor = (pixelRatio) => EDITOR_DASH_PX.map(length => length * pixelRatio);
// A handle's stroke in image pixels at the UNFLOORED ratio: its outline follows the handle (G8, D21; G9, D22), never
// thinner than 1 CSS px.
const handleOutlinePx = (handleRatio) => Math.max(handleRatio, LEVEL_HANDLE_OUTLINE_PX * handleRatio * handleScale(handleRatio));

// A level's overlay colour; the femoral heads share the femoral green.
function levelColor(code) {
  return kindOf(code) === 'circle' ? FEMORAL_OVERLAY_COLOR : LEVEL_RGB[code];
}

/**
 * The canvas colour a handle or a placement step is drawn in: SA SP IA IP are the corner colours, `trace` is
 * the rim-point colour, `level` is the level's own overlay colour (the femoral green for a head).
 */
export function colorFor(colorKey, code) {
  if (Object.hasOwn(CORNER_COLORS, colorKey)) return CORNER_COLORS[colorKey];
  if (colorKey === 'trace') return TRACE_COLOR;
  const rgb = levelColor(code);
  return rgb ? `rgb(${rgb.join(',')})` : STAGE_SELECTED_COLOR;
}

// A level's shape as drawable parts: a polyline (`line`, `closed`) or a circle (`arc`, `radius`). `area` parts
// can be filled; `dot` parts are the centroid marks, which stay filled when the level is selected. A two-point
// endplate has no area. Geometry that is not what its kind says gives no parts: a bad record draws nothing.
function shapeParts(code, geometry, pixelRatio) {
  if (!geometry) return [];
  switch (kindOf(code)) {
    case 'body': // (2026-10-21, partial endplates, spec §5) both pairs: the quadrilateral; one: that line alone
      // (2026-10-22, round 2, spec §12 N9) C7's optional centroid is a dot beside the outline, or alone
      return [
        ...(isPair(geometry.superior) && isPair(geometry.inferior) ? [{ line: quadrilateral(geometry), closed: true, area: true }]
          : isPair(geometry.superior) ? [{ line: geometry.superior }]
            : isPair(geometry.inferior) ? [{ line: geometry.inferior }] : []),
        ...(hasCentroid(code) && isPoint(geometry.centroid) ? [{ arc: geometry.centroid, radius: C2_CENTROID_RADIUS_PX * pixelRatio, area: true, dot: true }] : []),
      ];
    case 'c2': // (2026-10-21, partial endplates, spec §5) its line and/or its dot, whichever exists
      return [
        ...(isPair(geometry.inferior) ? [{ line: geometry.inferior }] : []),
        ...(isPoint(geometry.centroid) ? [{ arc: geometry.centroid, radius: C2_CENTROID_RADIUS_PX * pixelRatio, area: true, dot: true }] : []),
      ];
    case 'endplate':
      return isPair(geometry.superior) ? [{ line: geometry.superior }] : [];
    case 'centroid':
      return isPoint(geometry.centroid)
        ? [{ arc: geometry.centroid, radius: T4_CENTROID_RADIUS_PX * pixelRatio, area: true, dot: true }] : [];
    case 'circle':
      return isPoint(geometry.center) && geometry.radius >= 0
        ? [{ arc: geometry.center, radius: geometry.radius, area: true }] : [];
    default:
      return [];
  }
}

function tracePart(ctx, part) {
  ctx.beginPath();
  if (part.line) {
    part.line.forEach((point, index) => (index ? ctx.lineTo(point[0], point[1]) : ctx.moveTo(point[0], point[1])));
    if (part.closed) ctx.closePath();
  } else {
    ctx.arc(part.arc[0], part.arc[1], part.radius, 0, 2 * Math.PI);
  }
}

// The model's own geometry, dashed, under the current one.
function drawGhost(ctx, code, modelGeometry, pixelRatio) {
  ctx.save();
  ctx.setLineDash(dashFor(pixelRatio));
  ctx.strokeStyle = GHOST_COLOR;
  ctx.lineWidth = LEVEL_STROKE_PX * pixelRatio;
  for (const part of shapeParts(code, modelGeometry, pixelRatio)) {
    tracePart(ctx, part);
    ctx.stroke();
  }
  ctx.restore();
}

// A level's label plate (gate feedback G7, 2026-10-13; plan D20, rulings R43 and R44). `placement` is labelPlacement's
// (renderer/viewer/level-labels.js): the anchor, the posterior corner, and the side the plate extends to, away from the
// level, so the plate sits outside its body on the posterior side. Side +1: the plate's left edge LABEL_OFFSET_PX right
// of the anchor; side -1: its right edge LABEL_OFFSET_PX left of it. Centred on the anchor vertically, then moved by
// the level's drag offset (`offset`, image px, from state.platePositions). Returns the plate's rect in image pixels,
// after the offset, for the stage's hit test.
function drawLevelLabel(ctx, code, placement, offset, selected, pixelRatio, bounds) {
  const { anchor, side } = placement;
  ctx.font = `500 ${LABEL_FONT_PX * pixelRatio}px ${CANVAS_MONO}`;
  const width = ctx.measureText(code).width + 2 * LABEL_PAD_PX * pixelRatio;
  const height = LABEL_HEIGHT_PX * pixelRatio;
  const gap = LABEL_OFFSET_PX * pixelRatio;
  let x = (side < 0 ? anchor[0] - gap - width : anchor[0] + gap) + offset[0];
  let y = anchor[1] - height / 2 + offset[1];
  // (R45) The plate stays wholly on the film however far it is dragged, so it can always be seen and grabbed back;
  // a plate beside a body at the film's edge is pushed inward rather than clipped.
  if (bounds) {
    x = Math.min(Math.max(0, x), Math.max(0, bounds.width - width));
    y = Math.min(Math.max(0, y), Math.max(0, bounds.height - height));
  }
  ctx.fillStyle = LABEL_PLATE_FILL;
  if (typeof ctx.roundRect === 'function') {
    ctx.beginPath();
    ctx.roundRect(x, y, width, height, LABEL_RADIUS_PX * pixelRatio);
    ctx.fill();
  } else {
    ctx.fillRect(x, y, width, height);
  }
  ctx.fillStyle = selected ? STAGE_SELECTED_COLOR : STAGE_LABEL_FILL;
  ctx.textBaseline = 'middle';
  ctx.fillText(code, x + LABEL_PAD_PX * pixelRatio, y + height / 2);
  return { code, x, y, width, height };
}

// The selected level's draggable points (and a head's fixed centre dot, an R28 size at the floored `pixelRatio`). The
// focused handle also gets the ring; a focus that names another level, or a handle this level does not have, draws
// nothing extra. The disc, its outline and the ring are sized against `handleRatio`, the unfloored ratio (G9, D22).
function drawLevelHandles(ctx, code, geometry, focused, pixelRatio, handleRatio) {
  if (kindOf(code) === 'circle' && isPoint(geometry.center)) {
    ctx.beginPath();
    ctx.arc(geometry.center[0], geometry.center[1], CENTRE_DOT_RADIUS_PX * pixelRatio, 0, 2 * Math.PI);
    ctx.fillStyle = INK_COLOR;
    ctx.fill();
  }
  for (const { handle, point, colorKey } of levelHandles(code, geometry)) {
    if (focused && focused.code === code && focused.handle === handle) {
      ctx.beginPath();
      ctx.arc(point[0], point[1], HANDLE_RING_RADIUS_PX * handleRatio * handleScale(handleRatio), 0, 2 * Math.PI);
      ctx.strokeStyle = STAGE_SELECTED_COLOR;
      ctx.lineWidth = SELECTED_STROKE_PX * handleRatio;
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.arc(point[0], point[1], handleRadiusImagePx(handleRatio), 0, 2 * Math.PI);
    ctx.fillStyle = colorFor(colorKey, code);
    ctx.fill();
    ctx.strokeStyle = STAGE_BG_COLOR;
    ctx.lineWidth = handleOutlinePx(handleRatio);
    ctx.stroke();
  }
}

// One placed level: ghost, fill, outline, label. True when it was drawn (its handles are drawn by the caller, last).
// Its label plate's rect, when it has one, is pushed onto `plates`.
function drawPlacedLevel(ctx, editor, code, geometry, { overlays, fillAlpha, pixelRatio, anteriorSide, platePositions, plates, bounds }) {
  const selected = editor.selected === code;
  // Overlays off hides every level but the selected one: no fill, no outline, no label.
  if (!overlays && !selected) return false;
  const parts = shapeParts(code, geometry, pixelRatio);
  if (parts.length === 0) return false;
  if (selected && levelState({ ...editor.levels[code], geometry }, editor.model?.[code]) === 'edited') {
    drawGhost(ctx, code, editor.model[code], pixelRatio);
  }
  const rgb = levelColor(code);
  for (const part of parts) {
    tracePart(ctx, part);
    if (part.area && (part.dot || overlays)) {
      ctx.fillStyle = rgba(rgb, part.dot && selected ? 1 : fillAlpha);
      ctx.fill();
    }
    ctx.strokeStyle = selected ? STAGE_SELECTED_COLOR : rgba(rgb, LEVEL_STROKE_ALPHA);
    ctx.lineWidth = (selected ? SELECTED_STROKE_PX : LEVEL_STROKE_PX) * pixelRatio;
    ctx.stroke();
  }
  const placement = labelPlacement(code, geometry, anteriorSide);
  if (placement) plates.push(drawLevelLabel(ctx, code, placement, plateOffset(platePositions, code), selected, pixelRatio, bounds));
  return true;
}

// The placement in progress: each placed point as a dot in its step's colour, then either a dashed line through them
// in step order or, for a femoral head with two points or more, the dashed circle they fit and its centre. `points`
// holds one slot per step, null until set (gate feedback G2: the steps may be set in any order).
function drawPlacement(ctx, canvas, placing, pixelRatio, handleRatio) {
  const placed = (placing.points ?? []).flatMap((point, index) => (point ? [{ point, index }] : []));
  if (placed.length === 0) return;
  const points = placed.map(({ point }) => point);
  const { steps } = placementSteps(placing.code);
  const circle = kindOf(placing.code) === 'circle';
  const fit = circle && points.length >= 2 ? fitRim(points, Math.hypot(canvas.width, canvas.height)) : null;
  if (circle ? fit : points.length >= 2) {
    ctx.save();
    ctx.setLineDash(dashFor(pixelRatio));
    ctx.strokeStyle = STAGE_SELECTED_COLOR;
    ctx.lineWidth = PLACEMENT_LINE_PX * pixelRatio;
    ctx.beginPath();
    if (fit) {
      ctx.arc(fit.center[0], fit.center[1], fit.radius, 0, 2 * Math.PI);
    } else {
      points.forEach((point, index) => (index ? ctx.lineTo(point[0], point[1]) : ctx.moveTo(point[0], point[1])));
    }
    ctx.stroke();
    ctx.restore();
  }
  // The placed points are handles in size (G8, D21; G9, D22): the unfloored ratio, like the selected level's.
  placed.forEach(({ point, index }) => {
    ctx.beginPath();
    ctx.arc(point[0], point[1], handleRadiusImagePx(handleRatio), 0, 2 * Math.PI);
    ctx.fillStyle = colorFor(steps[index]?.colorKey ?? 'trace', placing.code);
    ctx.fill();
    ctx.strokeStyle = STAGE_BG_COLOR;
    ctx.lineWidth = handleOutlinePx(handleRatio);
    ctx.stroke();
  });
  if (fit) {
    ctx.beginPath();
    ctx.arc(fit.center[0], fit.center[1], CENTRE_DOT_RADIUS_PX * pixelRatio, 0, 2 * Math.PI);
    ctx.fillStyle = STAGE_SELECTED_COLOR;
    ctx.fill();
  }
}

/**
 * Draws the level editor (`state.levelEditor`) over the film. `opts`: `overlays` and `overlayOpacity` (the toolbar's
 * fill toggle and 0-100 slider), `pixelRatio` (image pixels per CSS pixel), `handle` (the focused handle,
 * `{code, handle}` or null) and `dragGeometry` (`{code, geometry}` or null: one level's geometry for this frame, a
 * drag in progress, in place of the stored one). Every visible level with geometry is drawn except the one being
 * placed; only the selected level has handles. (Gate feedback G7, 2026-10-13; plan D20) `anteriorSide` (the film's,
 * plateAnteriorSide in renderer/viewer/level-labels.js) is the plates' fallback side (R43), and `platePositions`
 * (`state.platePositions`, code -> [dx, dy] in image px) moves each dragged plate (R44).
 *
 * Returns the label plates it drew, `[{code, x, y, width, height}]` in image pixels after the offsets, in drawing
 * order (the last on top), so the stage can hit-test them without measuring text again. A caller that ignores it
 * is unaffected.
 */
export function drawLevelEditor(ctx, canvas, editor, opts) {
  const {
    overlays = true, overlayOpacity = 50, handle = null, dragGeometry = null, anteriorSide = null, platePositions = null,
  } = opts ?? {};
  // Ruling R28: the drawing ratio is floored at 1 image pixel per CSS pixel. The canvas is drawn at the stored
  // image's resolution and CSS-scaled with the film, so past 1:1 on screen (pixelRatio < 1, reachable on every film
  // since the zoom cap is 8x) a CSS-sized stroke, dash, handle or label would fall under one image pixel, be
  // anti-aliased away and then be magnified: faint lines, unreadable labels. Past 1:1 the overlay therefore keeps its
  // image-pixel size and magnifies with the film, crisp; below 1:1 it holds its size on screen, as before.
  const pixelRatio = Math.max(1, opts?.pixelRatio ?? 1);
  // (G9, D22) The handles alone take the UNFLOORED ratio, so past 1:1 they grow with the square root of the zoom
  // instead of magnifying with the film; a ratio that is not a positive number reads as 1:1, as the floored one does.
  const handleRatio = Number.isFinite(opts?.pixelRatio) && opts.pixelRatio > 0 ? opts.pixelRatio : 1;
  const fillAlpha = (BASE_OVERLAY_ALPHA / 255) * (Math.max(0, Math.min(100, overlayOpacity)) / 100);
  ctx.save();
  ctx.lineJoin = 'round';
  let selected = null;
  const plates = [];
  const level = { overlays, fillAlpha, pixelRatio, anteriorSide, platePositions, plates, bounds: { width: canvas.width, height: canvas.height } };
  for (const code of LEVEL_CODES) {
    const entry = editor.levels?.[code];
    if (!entry?.visible || editor.placing?.code === code) continue;
    const geometry = dragGeometry?.code === code ? dragGeometry.geometry : entry.geometry;
    if (!geometry) continue;
    if (drawPlacedLevel(ctx, editor, code, geometry, level) && editor.selected === code) {
      selected = { code, geometry };
    }
  }
  // The selected level's handles go above every level, so a later level's fill or outline never covers one.
  if (selected) drawLevelHandles(ctx, selected.code, selected.geometry, handle, pixelRatio, handleRatio);
  if (editor.placing) drawPlacement(ctx, canvas, editor.placing, pixelRatio, handleRatio);
  ctx.restore();
  return plates;
}

// Returns the label plates drawn (drawLevelEditor's return value) in edit mode, and none (an empty array) outside it,
// where no plate is drawn.
export function drawDynamicLayer(ctx, canvas, geometry, opts) {
  if (!opts?.editor) geometry = measurementGeometry(geometry, opts?.selectedLevel === 'GLOBAL_SVA' ? 'global_sva' : opts?.selectedLevel === 'PT' ? 'pelvic_tilt' : null);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (opts?.editor) {
    // (2026-10-10, level editor) The editor draws every level itself, so the region outlines, the S1 line, the C7
    // cross and the femoral circles are skipped. The selected measurement's construction stays
    // (under the editor), and a film with no geometry (a failed run) still has an editor to draw.
    if (geometry) drawSelectedMeasurement(ctx, canvas, geometry, opts.selectedLevel ?? null, opts.measurements);
    return drawLevelEditor(ctx, canvas, opts.editor, {
      overlays: opts.overlays ?? true,
      overlayOpacity: opts.overlayOpacity ?? 50,
      pixelRatio: opts.pixelRatio ?? 1,
      handle: opts.handle ?? null,
      dragGeometry: opts.dragGeometry ?? null,
      anteriorSide: opts.anteriorSide ?? null,
      platePositions: opts.platePositions ?? null,
    });
  }
  if (!geometry) return [];
  const selectedLevel = opts.selectedLevel ?? null;
  const lineWidth = Math.max(2, canvas.width / 600);
  ctx.lineJoin = 'round';
  for (const level of geometry.region === 'cervical' ? CERVICAL_LEVELS : geometry.region === 'full_spine' ? [...CERVICAL_LEVELS, ...LEVELS] : LEVELS) {
    const body = geometry.vertebrae?.[level];
    if (!body) continue;
    const selected = level === selectedLevel;
    ctx.strokeStyle = selected ? STAGE_SELECTED_COLOR
      : LEVEL_RGB[level] ? `rgb(${LEVEL_RGB[level].join(',')})` : STAGE_LINE_COLOR;
    ctx.lineWidth = selected ? lineWidth * 1.6 : lineWidth;
    ctx.beginPath();
    const outline = body.quadrilateral ?? body.inferior ?? body.superior; // (2026-10-21, partial endplates, spec §5)
    if (!outline) continue;
    outline.forEach((point, index) => (index ? ctx.lineTo(...point) : ctx.moveTo(...point)));
    if (body.quadrilateral) ctx.closePath();
    ctx.stroke();
    drawSelectedStageLabel(ctx, level, outline[0], selected, canvas.width);
  }
  const selectedS1 = selectedLevel === 'S1';
  ctx.strokeStyle = selectedS1 ? STAGE_SELECTED_COLOR : STAGE_LINE_COLOR;
  ctx.lineWidth = selectedS1 ? lineWidth * 1.6 : lineWidth;
  if (geometry.s1_superior) {
    ctx.beginPath();
    ctx.moveTo(...geometry.s1_superior[0]);
    ctx.lineTo(...geometry.s1_superior[1]);
    ctx.stroke();
    drawSelectedStageLabel(ctx, 'S1', geometry.s1_superior[0], selectedS1, canvas.width);
  }
  // (2026-10-22, spec P11/P12) The T1 superior endplate as a line in T1's colour at the S1 line's width, on every region
  // (a cervical film has no S1 line but has this one), labelled while the T1 slope is selected.
  if (isPair(geometry.t1_superior)) {
    const t1 = geometry.t1_superior;
    const selectedT1 = selectedLevel === 'T1S';
    ctx.strokeStyle = `rgb(${LEVEL_RGB.T1.join(',')})`;
    ctx.lineWidth = selectedT1 ? lineWidth * 1.6 : lineWidth;
    ctx.beginPath();
    ctx.moveTo(...t1[0]);
    ctx.lineTo(...t1[1]);
    ctx.stroke();
    drawSelectedStageLabel(ctx, 'T1', t1[0], selectedT1, canvas.width);
  }
  // A centroid cross in the level's colour: C7's on a full-spine film (the C7–S1 plumb's origin, labelled while
  // GLOBAL_SVA is selected) and, since gate feedback G11 (2026-10-16, decision D24), C2's on a cervical or full-spine
  // film (the C2–C7 plumb's origin, labelled while C2C7_SVA is selected), so both ends of the cervical plumb read
  // alike outside edit mode; before G11 the C2 centroid was drawn only in the editor, as a dot.
  const crossArm = Math.max(4, canvas.width / 150);
  const drawCentroidCross = (point, level, selected) => {
    if (!isPoint(point)) return;
    const [x, y] = point;
    ctx.strokeStyle = `rgb(${LEVEL_RGB[level].join(',')})`;
    ctx.lineWidth = lineWidth;
    ctx.beginPath(); ctx.moveTo(x - crossArm, y); ctx.lineTo(x + crossArm, y);
    ctx.moveTo(x, y - crossArm); ctx.lineTo(x, y + crossArm); ctx.stroke();
    drawSelectedStageLabel(ctx, level, point, selected, canvas.width);
  };
  if (geometry.region === 'full_spine') drawCentroidCross(geometry.c7_centroid, 'C7', selectedLevel === 'GLOBAL_SVA');
  if (['cervical', 'full_spine'].includes(geometry.region)) drawCentroidCross(geometry.c2_centroid, 'C2', selectedLevel === 'C2C7_SVA');
  // (2026-10-22, spec P11) T4's centroid cross, on every region, labelled while the T4 pelvic angle is selected.
  drawCentroidCross(geometry.t4_centroid, 'T4', selectedLevel === 'T4PA');

  // Ruling R28, as in drawLevelEditor: past 1:1 the hip line, its dash and the markers keep their image-pixel size
  // and magnify with the film instead of shrinking under one image pixel; below 1:1 they hold their screen size.
  const pixelRatio = Math.max(1, opts.pixelRatio ?? 1);
  (geometry.femoral_circles ?? []).forEach(([x, y, r], index) => {
    const selectedCircle = Boolean(opts.editing) && opts.selection?.kind === 'femoral'
      && opts.selection.side === FEMORAL_SIDES[index];
    ctx.strokeStyle = selectedCircle ? STAGE_SELECTED_COLOR : FEMORAL_HANDLE_COLOR;
    ctx.lineWidth = selectedCircle ? lineWidth * 1.6 : lineWidth;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.stroke();
    if (!opts.editing) {
      const arm = 4 * pixelRatio;
      ctx.beginPath();
      ctx.moveTo(x - arm, y); ctx.lineTo(x + arm, y);
      ctx.moveTo(x, y - arm); ctx.lineTo(x, y + arm);
      ctx.stroke();
    }
  });
  if (geometry.femoral_circles?.length === 2 && geometry.hip_midpoint) {
    const [a, b] = geometry.femoral_circles;
    const [x, y] = geometry.hip_midpoint;
    ctx.save();
    ctx.strokeStyle = FEMORAL_HANDLE_COLOR;
    ctx.lineWidth = 1.5 * pixelRatio;
    ctx.setLineDash([4 * pixelRatio, 3 * pixelRatio]);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    ctx.setLineDash([]);
    // A diamond distinguishes the bilateral midpoint from each circle's centre handle.
    const arm = 5 * pixelRatio;
    ctx.beginPath(); ctx.moveTo(x, y - arm); ctx.lineTo(x + arm, y);
    ctx.lineTo(x, y + arm); ctx.lineTo(x - arm, y); ctx.closePath();
    ctx.fillStyle = STAGE_BG_COLOR; ctx.fill(); ctx.stroke();
    ctx.restore();
  }

  drawSelectedMeasurement(ctx, canvas, geometry, selectedLevel, opts.measurements);
  return [];
}
