let state = {
  screen: 'landing',
  ack: false,
  theme: 'light',
  navCollapsed: false,
  settingsOpen: false,

  studies: [],
  // Unmeasured edits are session-only. Study.geometry/measurements always form
  // a complete pair, including when the saver runs or the app closes mid-edit.
  measurementDrafts: {},
  deletingStudies: false,
  query: '',
  // The Studies screen's second tab (pre-op/post-op spec §10). Which tab is up, and the
  // Parameters tab's filters, sort and lordosis-level toggle. Store state rather than screen
  // scope so coming back from Analysis lands on the tab, filters and sort the user left. Each
  // is replaced wholesale on change -- the screen's gate compares by reference.
  studiesTab: 'find',
  paramFilters: { workspace: null, folder: null, segmentedOnly: true, timepoint: null, view: null, subject: '', pairedOnly: false, pairedWith: '__any__' },
  paramSort: { key: 'study', dir: 'asc' },
  paramLevels: false,
  // The study ids ticked on the Parameters grid (addendum, 2026-09-07); replaced wholesale;
  // session-only, never persisted.
  paramSelected: [],
  // (2026-09-10, studies-table spec 6.1) the Find list's sort: key 'study'|'subject'|'region'|'view'|'workspace'|
  // 'folder'|'date'|'status', dir 'asc'|'desc'. Replaced wholesale on change; session-only, never persisted;
  // read by the Studies screen's own subscription, never by SCREEN_KEYS. The grid's paramSort is separate.
  findSort: { key: 'date', dir: 'desc' },
  // (2026-10-19, header filters; spec F6) the Find list's own per-column ticks, one list per header menu plus the
  // SUBJECT menu's text (data/find.js DEFAULT_FIND_FILTERS). Replaced wholesale; session-only, never persisted; written
  // by the Studies screen's header menus and read through data/find.js by its rows and the Analysis screen's
  // previous/next, never by SCREEN_KEYS. It replaced the list's share of paramFilters (workspace, folder) and the
  // reviewer navigation round's single status filter, whose job the STATUS menu took over. The grid keeps paramFilters
  // and its own segmentedOnly.
  findFilters: { subject: '', region: [], view: [], workspace: [], folder: [], status: [] },
  openId: null,
  compareId: null,

  tab: 'meas',
  selectedLevel: null,
  overlays: true,
  overlayOpacity: 50,
  zoom: 1,
  panX: 0,
  panY: 0,
  panMode: false,
  showAllLordosis: false,

  // Which model reads which structure on the next run; see renderer/data/models.js.
  models: { vertebrae: 'hrnet', femoral: 'unet', s1: 'keypointrcnn' },
  performance: { mode: 'standard', cpuThreads: 2, cropLocalizer: true, cropMethod: 'search', viewSelection: 'landmarks', toolbarRemoval: false, processor: 'cpu' },
  // [{id, kind, name}] from the backend's GET /processors, CPU first; null until listed. Session only.
  processors: null,

  editing: false,
  selection: null,
  // (2026-10-10, level editor) The edit-mode level editor's state, or null outside edit mode; see the LevelEditor
  // typedef in renderer/data/levels.js. Written only by renderer/level-editor.js; session-only, never persisted.
  levelEditor: null,
  // (2026-10-11, gate feedback G3; plan D15) The open film's window / level, or null: display only, part of the view
  // like zoom and pan. FRESH_VIEW resets it; entering and leaving edit mode keep it. See the FilmWindow typedef in
  // renderer/film-window.js, its only writer. Session-only, never persisted, never in the label record.
  filmWindow: null,
  // (2026-10-13, gate feedback G7; plan D20, ruling R44) Where the user has dragged each level's label plate in edit
  // mode: level code -> [dx, dy] in image pixels. View state like zoom and pan: FRESH_VIEW resets it, entering and
  // leaving edit mode keep it. Written only by components/viewer.js, always as a new object; session-only, never
  // persisted, never in the label record.
  platePositions: {},
  running: null, // string|null — the id of the study whose /predict is in flight; one run at a time
  runStage: null,
  // (2026-09-08, batch spec 8.1) The running batch, or null: { ids, done, failed, warnings, skipped,
  // stopping }. Replaced wholesale on every change by renderer/batch.js; never persisted.
  batch: null,

  calibrationRequest: null, // folder-upload handoff; session only
  wsFolder: null,
  wsFiles: [],
  wsCsv: null,
  wsCsvHeaders: [],
  wsCsvRows: [],
  wsMapping: [],

  // (2026-09-07, pre-op/post-op spec §8.5) One row per scanned folder that holds films --
  // { folder, count, timepoint, view } -- the assignment the Workspace card shows before Load.
  // Rebuilt by every scan, replaced wholesale on every change, never persisted.
  wsFolderRows: [],

  fields: [],
  dataOpen: true,
  toast: '',
};

const listeners = new Set();

// True only while setState is iterating listeners for the current update.
// Enforces the architecture contract's rule that subscribers must not call
// setState during notification (see setState below).
let notifying = false;

export function getState() {
  return Object.freeze({ ...state });
}

export function setState(patchOrFn) {
  if (notifying) {
    throw new Error(
      'setState() must not be called from a subscriber; the contract forbids re-entrant updates.'
    );
  }
  const patch = typeof patchOrFn === 'function' ? patchOrFn(state) : patchOrFn;
  state = { ...state, ...patch };
  notifying = true;
  try {
    for (const listener of listeners) {
      try {
        listener(state);
      } catch (error) {
        // One subscriber's throw must not silence the subscribers after it. Before this
        // guard, a TypeError inside a canvas draw function stopped the router's listener
        // too, and the whole UI froze instead of one layer going blank. The re-entrancy
        // error thrown by the nested setState() above is unaffected: it is raised inside
        // the offending subscriber, which is where it belongs.
        console.error('store: subscriber threw during notification', error);
      }
    }
  } finally {
    notifying = false;
  }
}

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
