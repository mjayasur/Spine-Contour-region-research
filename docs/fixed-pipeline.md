# Fixed measurement pipeline

The production configuration is defined by `backend/pipeline_stages.py` and
`backend/fixed_pipeline.py`. Stage replacement is a code/configuration change;
users do not select experimental combinations in the application.

| Measurements | Predictor | Decoder | Lumbar candidate policy |
| --- | --- | --- | --- |
| C2–C7 Cobb, C2–C7 SVA | Cervical slope HRNet, epoch 12 | 9×9 local softmax, temperature 12 | Automatic neck crop |
| Global SVA | Anatomical cervical HRNet, epoch 87; lumbar HRNet, epoch 45 | Original | Original and 15% upward extension |
| L1–S1 lordosis, sacral slope, pelvic incidence | Lumbar HRNet, epoch 45 | 9×9 local softmax, temperature 12 | Every candidate extended upward 15% |
| Pelvic tilt | Lumbar HRNet, epoch 45 | 9×9 local softmax, temperature 12 | Original and 15% upward extension |

Pelvic measurements use the conservative femoral U-Net checkpoint. S1 localization
and corroboration use the original Keypoint R-CNN. Existing cervical localization
uses DETR and image-sided HRNet crop agreement. Those localization models do not
replace the anatomical measurement predictors in the table.

## Stage contracts

`Stages` supplies replaceable lumbar and cervical locators, cervical candidate
generation, S1 detection, femoral segmentation, and predictor graph identifiers.
Lumbar locators receive a grayscale image in canonical anterior-left orientation
and return a crop record with a `window` or `None`. Rectangles use half-open
`left, top, right, bottom` pixel coordinates. Cervical regional locators return
an original-image rectangle. Model coordinates are restored once, before
measurement. No stage receives reader measurements or reference landmarks.

`measurement_sources` retains the distinct global-SVA and pelvic-tilt anchors.
The viewer and measurement service select the same source. Human changes to
shared S1/hip/C7 anchors supersede the affected automatic source; reset restores
the complete original geometry. Source-image scale is supplied independently.

## NVIDIA application runtime

Install `backend/requirements-cuda.txt` in a separate environment, instead of
`backend/requirements.txt`. This is an application runtime, not a benchmark-only
adapter. The app's processor menu discovers NVIDIA devices by UUID and passes
the selected device to the same inference code used by the validation runner.
The ordinary DirectML and macOS CPU runtimes remain supported.

CUDA sessions budget 92% of currently free device memory. Dynamic HRNet batches
start with all available crop candidates, halve after allocation failure, and
retry the same rows in order. A one-row failure reaches the request boundary,
which releases model sessions and retries the whole film on CPU. Other GPU
execution errors also reach that boundary. Cancellation is checked between
batches. Batch sizes and OOM retries are recorded in `qc.processing.batches`.

An optional four-image S1 graph accelerates detector calls; if it cannot fit,
the same checkpoint runs one crop at a time on GPU. The graph is exported with
`tools/export_s1_batch.py`. Neither batching nor recovery changes image size,
crop policy, checkpoints, or decoders. NVIDIA TF32 convolution behavior matches
the reference CUDA evaluation; exported graph tensors are float32.

## Reproduction

`tools/reproduce_fixed_pipeline.py` runs the application stages on a private
manifest, checks image hashes and saves geometry, provenance and timings.
Frozen-localization and `--fresh` runs are distinct. Expected workbook values
are read only by `tools/compare_fixed_pipeline.py`, after inference. The default
comparison requires identical missing-value availability and a maximum absolute
difference of 0.1 degrees or millimetres. Calibration is an explicit input;
using frozen scale does not validate automatic ruler extraction.
