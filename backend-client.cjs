const http = require('node:http');

// 'cpu', or a GPU's PCI identity as backend/processors.py names it. The same pattern is in
// renderer/data/processing.js, which cannot require this file.
const PROCESSOR_ID = /^(?:cpu|cuda:GPU-[0-9a-fA-F-]+|gpu:[0-9a-f]{4,8}:[0-9a-f]{4,8}(?::(?:[2-9]|[1-9][0-9]))?)$/;

function normalizePerformance(value) {
  const mode = value?.mode ?? 'standard';
  const cpuThreads = value?.cpuThreads ?? 2;
  const cropLocalizer = value?.cropLocalizer === undefined ? true : value.cropLocalizer;
  const viewSelection = value?.viewSelection === undefined ? 'landmarks' : value.viewSelection;
  const cropMethod = value?.cropMethod === undefined ? 'search' : value.cropMethod;
  const toolbarRemoval = value?.toolbarRemoval === undefined ? false : value.toolbarRemoval;
  const processor = value?.processor === undefined ? 'cpu' : value.processor;
  if (!['standard', 'low-memory'].includes(mode) || !Number.isInteger(cpuThreads) || cpuThreads < 1 || cpuThreads > 4
    || typeof cropLocalizer !== 'boolean' || !['search', 'model'].includes(cropMethod)
    || !['landmarks', 'classifier'].includes(viewSelection)
    || typeof toolbarRemoval !== 'boolean'
    || typeof processor !== 'string' || !PROCESSOR_ID.test(processor)) {
    throw new Error('Invalid processing settings.');
  }
  // Migrate the retired trained-crop preference for saved settings and all requests.
  return { mode, cpuThreads, cropLocalizer, cropMethod: 'search', toolbarRemoval, processor, viewSelection };
}

// The backend's GET /processors body, reduced to what Settings shows. Anything malformed is
// dropped, and the CPU is always offered first.
function normalizeProcessors(body) {
  const listed = Array.isArray(body?.processors) ? body.processors : [];
  const gpus = listed.filter((item) => item?.kind === 'gpu' && typeof item.id === 'string' && item.id !== 'cpu'
    && PROCESSOR_ID.test(item.id) && typeof item.name === 'string' && item.name.trim());
  return [{ id: 'cpu', kind: 'cpu', name: 'CPU' },
    ...gpus.map((item) => ({ id: item.id, kind: 'gpu', name: item.name.trim() }))];
}

// Node's default fetch header deadline is unsuitable for long local inference.
// The streaming route sends a heartbeat every two seconds. There is no total
// job deadline; an idle/dead connection still fails, and cancellation closes it.
async function postForm(url, form, { signal, onProgress, idleMs = 60000 } = {}) {
  const encoded = new Request(url, { method: 'POST', body: form });
  const bytes = Buffer.from(await encoded.arrayBuffer());
  if (signal?.aborted) throw new Error('Processing cancelled.');
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (error, result) => {
      if (settled) return;
      settled = true;
      signal?.removeEventListener('abort', cancel);
      error ? reject(error) : resolve(result);
    };
    const request = http.request(url, {
      method: 'POST', headers: { 'content-type': encoded.headers.get('content-type'), 'content-length': bytes.length },
    }, async (response) => {
      response.setEncoding('utf8');
      let buffer = '';
      try {
        const streaming = response.headers['content-type']?.includes('application/x-ndjson');
        for await (const chunk of response) {
          buffer += chunk;
          if (!streaming) continue;
          let newline;
          while ((newline = buffer.indexOf('\n')) >= 0) {
            const line = buffer.slice(0, newline);
            buffer = buffer.slice(newline + 1);
            if (!line.trim()) continue;
            const event = JSON.parse(line);
            if (event.type === 'result') { finish(null, event.result); return; }
            if (event.type === 'error') throw new Error(event.message || 'Processing failed.');
            if (event.type === 'progress' || event.type === 'heartbeat') onProgress?.(event);
          }
        }
        if (streaming) throw new Error('Processing connection ended before a result was received.');
        const result = JSON.parse(buffer);
        if (response.statusCode < 200 || response.statusCode >= 300) {
          throw new Error(typeof result.detail === 'string' ? result.detail : `Processing failed (${response.statusCode}).`);
        }
        finish(null, result);
      } catch (error) { finish(error); }
    });
    const cancel = () => request.destroy(new Error('Processing cancelled.'));
    signal?.addEventListener('abort', cancel, { once: true });
    request.setTimeout(idleMs, () => request.destroy(new Error('The processing worker stopped responding.')));
    request.on('error', (error) => finish(error));
    request.end(bytes);
  });
}

module.exports = { postForm, normalizePerformance, normalizeProcessors };
