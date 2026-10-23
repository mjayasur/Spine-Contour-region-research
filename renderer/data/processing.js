export const DEFAULT_PERFORMANCE = Object.freeze({ mode: 'standard', cpuThreads: 2, cropLocalizer: true, cropMethod: 'search', viewSelection: 'landmarks', toolbarRemoval: false, processor: 'cpu' });

// 'cpu', or a GPU's PCI identity as backend/processors.py names it (mirrors backend-client.cjs).
const PROCESSOR_ID = /^(?:cpu|cuda:GPU-[0-9a-fA-F-]+|gpu:[0-9a-f]{4,8}:[0-9a-f]{4,8}(?::(?:[2-9]|[1-9][0-9]))?)$/;

export function validPerformance(value) {
  return value && ['standard', 'low-memory'].includes(value.mode)
    && Number.isInteger(value.cpuThreads) && value.cpuThreads >= 1 && value.cpuThreads <= 4
    && typeof value.cropLocalizer === 'boolean' && value.cropMethod === 'search'
    && ['landmarks', 'classifier'].includes(value.viewSelection === undefined ? 'landmarks' : value.viewSelection)
    && typeof value.toolbarRemoval === 'boolean'
    && typeof value.processor === 'string' && PROCESSOR_ID.test(value.processor);
}

// The Settings choices: what the backend listed (null until it answers), and a saved GPU it no
// longer finds, kept visible so the select never shows a choice the user did not make. A run
// with a missing GPU uses the CPU and says so.
export function processorChoices(processors, selected) {
  const listed = Array.isArray(processors) ? processors : [];
  const choices = [{ id: 'cpu', label: 'CPU', missing: false },
    ...listed.filter((item) => item.kind === 'gpu').map((item) => ({ id: item.id, label: `GPU · ${item.name}`, missing: false }))];
  if (!choices.some((choice) => choice.id === selected)) {
    choices.push({ id: selected, label: processors ? 'Saved GPU · not found' : 'Saved GPU', missing: Boolean(processors) });
  }
  return choices;
}

// Where a stored result's models ran, from the per-model providers the backend recorded
// (`qc.processing.providers`): 'GPU' when every model used DirectML, 'CPU' when none did, and
// 'GPU + CPU' when some fell back. Records from before the processor setting carry no
// `processor` and read null: absent, never a guess.
export function describeProcessor(qc) {
  const processing = qc?.processing;
  if (!processing?.processor) return null;
  const lists = Object.values(processing.providers ?? {}).filter(Array.isArray);
  if (lists.length === 0) return null;
  const onGpu = lists.filter((providers) => providers.includes('DmlExecutionProvider')).length;
  return onGpu === 0 ? 'CPU' : onGpu === lists.length ? 'GPU' : 'GPU + CPU';
}

// The tooltip beside `describeProcessor`: which GPU ran, or why the run was not on the one chosen.
export function processorTitle(qc) {
  const where = describeProcessor(qc);
  if (!where) return '';
  const record = qc.processing.processor;
  if (typeof record.note === 'string' && record.note) return record.note;
  if (where === 'CPU') return 'The models ran on the CPU';
  return `The models ran on ${record.name}${where === 'GPU' ? '' : '; some fell back to the CPU'}`;
}

// The Settings note under the processor choice. `processors` is null until the backend lists
// them (or if it could not), so an unknown list is never described as empty or in progress.
export function processorNote(processors, selected) {
  const gpus = Array.isArray(processors) ? processors.filter((item) => item.kind === 'gpu') : null;
  if (selected !== 'cpu') {
    if (!gpus) return 'Runs the models on the selected GPU, or on the CPU if it is not found.';
    // The select clips long adapter names; the note carries the whole name.
    const gpu = gpus.find((item) => item.id === selected);
    return gpu
      ? `Runs the models on ${gpu.name} through DirectML. Checks CPU parity before first use. If GPU processing fails, the entire film restarts on the CPU.`
      : 'This GPU was not found. Runs use the CPU until it is back or another is chosen.';
  }
  if (!gpus) return 'Runs the models on the CPU.';
  return gpus.length ? 'Runs the models on the CPU. Choose a GPU to run them there instead.'
    : 'CPU processing is available. GPU acceleration requires a supported Windows DirectX 12 device.';
}

export function progressUpdate(current, event) {
  if (!current || current.requestId !== event?.requestId || current.cancelling || current.stage === 'saving') return current;
  if (!['progress', 'heartbeat'].includes(event.type)) return current;
  const elapsed = Number.isFinite(event.elapsed_seconds) && event.elapsed_seconds >= 0
    ? Math.max(current.elapsed_seconds ?? 0, event.elapsed_seconds) : current.elapsed_seconds;
  if (event.type === 'heartbeat') return { ...current, elapsed_seconds: elapsed };
  if (typeof event.message !== 'string' || !event.message.trim()) return current;
  const hasCount = Number.isInteger(event.completed) && Number.isInteger(event.total)
    && event.total > 0 && event.completed >= 0 && event.completed <= event.total;
  return { ...current, stage: event.stage, message: event.message, elapsed_seconds: elapsed,
    completed: hasCount ? event.completed : null, total: hasCount ? event.total : null };
}

export function progressTitle(progress) {
  return progress?.cancelling ? 'Cancelling processing…' : progress?.message || 'Starting image processing…';
}

export function progressDetail(progress) {
  if (!progress) return 'Waiting for the processing worker';
  if (progress.cancelling) return 'Stopping after the current operation. Completed results are kept.';
  const seconds = Math.max(0, Math.floor(progress.elapsed_seconds || 0));
  const elapsed = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')} elapsed`;
  const parts = [];
  if (progress.total > 0) {
    const units = progress.stage === 'search' ? 'regions checked' : 'passes completed';
    parts.push(`${progress.completed} of ${progress.total} ${units}`);
  }
  parts.push(elapsed);
  if (progress.mode === 'low-memory') parts.push('Low memory');
  return parts.join(' · ');
}
