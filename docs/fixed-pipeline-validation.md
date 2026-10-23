# Fixed-pipeline validation

## Measurement reproduction

The cohort contains 99 real full-spine radiographs. Each input file is checked
against the private manifest's SHA-256. Inference uses the specified checkpoint,
decoder, and crop policy for each measurement. Image-derived cervical windows,
lumbar localizers, orientation, and pixel spacing are frozen to the reference
evaluation. No workbook value or reader landmark enters inference or candidate
selection. Comparison runs separately after inference.

Workbook SHA-256: `ee4496f1fc0cdfc81a70540964059cac48e366574e39b85af397da4900b4def8`.

All seven measurements pass a maximum absolute difference of 0.1 degrees or
millimetres, with identical measurement availability for every image.

| Measurement | Numeric pairs | Both unavailable | Maximum absolute difference (° or mm) |
| --- | ---: | ---: | ---: |
| C2-C7 Cobb | 89 | 10 | 0.01617822 |
| C2-C7 SVA | 83 | 16 | 0.0028906928 |
| Global SVA | 87 | 12 | 6.3948846e-14 |
| L1-S1 Lordosis | 95 | 4 | 0.0059674478 |
| Sacral Slope | 95 | 4 | 0.0064458007 |
| Pelvic Incidence | 89 | 10 | 0.0063788044 |
| Pelvic Tilt | 93 | 6 | 0.0012819733 |

Global SVA uses the original lumbar graph including its embedded decoder.
Decoding a separately exported heatmap can alter the medoid among closely
spaced crop candidates; preserving the original graph is part of the recipe.
Cervical subpixel neighborhoods include each in-bounds heatmap cell once.
The lumbar decoder retains its reference boundary-clipping convention.

## Runtime and memory

Hardware: RTX 3090 Ti and RTX 3090, 24 GB each. Other tower processes remained
running. CUDA validation used ONNX Runtime 1.23.2, float32 graph tensors, and
the reference TF32 convolution setting. CUDA/cuDNN libraries were supplied by
the tower environment. The separate CUDA requirements file pins this tested
runtime. A packaged Windows CUDA installation has not been validated.

The 50-image shard took 326.96 seconds; the 49-image shard took 322.10 seconds.
The longest shard is **5 minutes 27 seconds**. This timing includes image
loading, prediction, measurement and JSON output, but excludes fresh localization,
automatic calibration, HTTP encoding and application rendering. It is not an
end-to-end 99-image app benchmark.

Two real-image HTTP `/predict` checks with fresh localization and automatic
calibration returned HTTP 200 on the RTX 3090 in 70.41 and 63.36 seconds. Both
remained on CUDA and used four-crop S1 batches and 14-crop lumbar heatmap batches.
Automatic ruler extraction returned `not_found` for both, so physical SVA values
require a saved or reviewed scale in those app results. The workbook comparison
uses the reference evaluation's scale.

A real CUDA allocation-pressure probe limited the lumbar graph's arena to
1,024 MiB. Allocation failures reduced batches from 14 to 7 to 3, and a later
two-crop remainder reduced to one. All 14 crops completed in order. The largest
decoded-point difference from single-crop CUDA inference was 0.003632 model
pixels. Heatmaps were not bit-identical: their maximum difference was 0.005098.
Unit checks additionally cover cancellation boundaries, non-allocation errors,
single-row failure, session eviction and whole-request CPU fallback.

## Application and geometry checks

The backend suite passed 808 tests with 3 platform/data-dependent skips; the
renderer suite passed 1,252 tests. API contracts cover saved calibration,
streaming, partial anatomy, automatic region selection, fixed model choices,
and source-coordinate preservation. Separate geometry tests check that the
PT and global-SVA sources survive recomputation and that edits supersede the
affected automatic anchors.

Predicted overlays were visually inspected on three real cohort radiographs,
including a postoperative lumbar film and an obscured lower cervical region.
The inspection checked source-coordinate placement and measurement-specific
crop geometry. It is not a blinded anatomical-accuracy study or an assessment
of all 99 overlays. Radiograph pixels and case-level measurements remain private.

## Remaining acceptance criteria

The requested **3–5 minute end-to-end cohort target is not met**. A complete
99-image run with fresh localization, automatic scale extraction and app output
has not been validated. The detector search dominates fresh-image time. CUDA
batching and allocation recovery operate in the shared app backend, but they
do not establish the requested end-to-end throughput. The production HTTP API
supports the fixed recipe; legacy internal inference helpers remain for offline
compatibility and can be removed in a separate, measured cleanup.

## Repeating the checks

Use `tools/extract_measurement_reference.py` to extract the workbook's AI column.
Keep the resulting reference JSON separate from the inference manifest. Each
manifest row contains `case`, `image`, `sha256`, `mm_per_pixel` and `localization`.
Localization contains `anterior_side`, a source-coordinate `cervical_window`,
and `lumbar_localizer` with a canonical anterior-left `window`.

Run `tools/reproduce_fixed_pipeline.py` with `--manifest`, `--output`, `--gpu`,
`--shard` and `--shards`. Add `--fresh` to recompute localization; this runner
still uses manifest calibration. `tools/compare_fixed_pipeline.py` requires
`--reference`, `--predictions` and `--output` and exits nonzero on missing cases,
availability changes or excessive differences. `tools/smoke_fixed_api.py`
exercises the actual HTTP route with fresh localization and OCR.
`tools/probe_cuda_oom.py` exercises allocation backoff with real crop tensors.
