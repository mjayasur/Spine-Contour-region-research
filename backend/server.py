"""FastAPI server for Spine-Contour inference."""

from __future__ import annotations

import base64
import io
import json
import hashlib
import logging

import numpy as np
import onnxruntime as ort
import pydicom
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

try:
    from . import processors, runtime
    from .progress import stream_job
    from .models.models import model_versions, release_models
    from .decode import decode_film
    from .calibration import calibration_from_payload, learn_profile, validate_profile
    from .cervical_measurements import cervical_measurements_from_geometry
    from .measurement_routes import measure as global_sva_measurements_from_geometry
    from .thoracic_measurements import thoracic_measurements
    from .utils import (
        spinopelvic_measurements_from_geometry,
    )
except ImportError:  # Support `uvicorn server:app` from backend/.
    import processors
    import runtime
    from progress import stream_job
    from models.models import model_versions, release_models
    from decode import decode_film
    from calibration import calibration_from_payload, learn_profile, validate_profile
    from cervical_measurements import cervical_measurements_from_geometry
    from measurement_routes import measure as global_sva_measurements_from_geometry
    from thoracic_measurements import thoracic_measurements
    from utils import (
        spinopelvic_measurements_from_geometry,
    )


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
CERVICAL_MODEL = "cervical_slope_hrnet"
FULL_SPINE_MODEL = "dual_hrnet"

app = FastAPI(title="Spine-Contour", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


def _dicom_pixel_array(payload: bytes) -> np.ndarray:
    dataset = pydicom.dcmread(io.BytesIO(payload))
    pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
    if pixels.ndim != 2:
        raise ValueError("Only single-frame grayscale DICOM files are supported")
    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    pixels = pixels * slope + intercept
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        pixels = float(pixels.max() + pixels.min()) - pixels
    return pixels


def _decode_grayscale(payload: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            # Preserve native grayscale precision until the model's percentile
            # rescale. PIL's conversion to L clips 16-bit values above 255.
            if image.mode in ("I", "F") or image.mode.startswith("I;16"):
                return np.array(image)
            return np.asarray(image.convert("L"))
    except (UnidentifiedImageError, OSError):
        try:
            return _dicom_pixel_array(payload)
        except Exception as error:
            raise ValueError("The upload is not a readable image or grayscale DICOM file") from error


def cervical_prediction(pixel_array, anterior_side, model=CERVICAL_MODEL):
    from .fixed_pipeline import predict
    return predict(pixel_array, anterior_side, region="cervical")


def full_spine_prediction(pixel_array, anterior_side, model=FULL_SPINE_MODEL, **options):
    # Existing regional inference does not import or initialize this pipeline.
    if __package__:
        from .fixed_pipeline import predict as predict_full_spine
    else:
        from fixed_pipeline import predict as predict_full_spine
    return predict_full_spine(pixel_array, anterior_side, **options)


def detect_film(pixel_array, anterior_side=None):
    if runtime.options().view_selection == "classifier":
        if __package__:
            from .view_classifier import detect_film as fast_detect
        else:
            from view_classifier import detect_film as fast_detect
        return fast_detect(pixel_array, anterior_side)
    if __package__:
        from .film_detection import detect_film as detect
    else:
        from film_detection import detect_film as detect
    return detect(pixel_array, anterior_side=anterior_side)


def _validate_auto_request(modality, view, laterality, vertebra_model,
                           femoral_model, s1_model, anterior_side):
    normalize = lambda value: (value or "").strip().lower().replace("-", "").replace("_", "")
    normalized_view, normalized_laterality = normalize(view), normalize(laterality)
    if normalized_view and normalized_laterality and normalized_view != normalized_laterality:
        raise ValueError("view and laterality must agree when both are provided")
    if normalize(modality) != "xray" or (normalized_view or normalized_laterality) != "lateral":
        raise ValueError("Automatic film detection requires modality='xray', view='lateral'")
    if anterior_side not in (None, "", "auto", "left", "right"):
        raise ValueError("Anterior side must be auto, left or right")
    if vertebra_model or femoral_model or s1_model:
        raise ValueError("Choose a spine region before overriding its models")


def _validate_cervical_request(modality, view, laterality, vertebra_model,
                               femoral_model, s1_model, anterior_side):
    normalize = lambda value: (value or "").strip().lower().replace("-", "").replace("_", "")
    normalized_view, normalized_laterality = normalize(view), normalize(laterality)
    if normalized_view and normalized_laterality and normalized_view != normalized_laterality:
        raise ValueError("view and laterality must agree when both are provided")
    if normalize(modality) != "xray" or (normalized_view or normalized_laterality) != "lateral":
        raise ValueError("Cervical models require modality='xray', body_part='cervical', view='lateral'")
    if anterior_side not in (None, "", "auto", "left", "right"):
        raise ValueError("Cervical anterior side must be auto, left or right")
    if vertebra_model not in (None, "", CERVICAL_MODEL):
        raise ValueError(f"Unknown cervical vertebra model; available: {CERVICAL_MODEL}")
    if femoral_model or s1_model:
        raise ValueError("Femoral and S1 models are not available for cervical radiographs")


def _validate_full_spine_request(modality, view, laterality, vertebra_model,
                                 femoral_model, s1_model, anterior_side):
    normalize = lambda value: (value or "").strip().lower().replace("-", "").replace("_", "")
    normalized_view, normalized_laterality = normalize(view), normalize(laterality)
    if normalized_view and normalized_laterality and normalized_view != normalized_laterality:
        raise ValueError("view and laterality must agree when both are provided")
    if normalize(modality) != "xray" or (normalized_view or normalized_laterality) != "lateral":
        raise ValueError("Global SVA requires modality='xray', body_part='full_spine', view='lateral'")
    if anterior_side not in (None, "", "auto", "left", "right"):
        raise ValueError("Full-spine anterior side must be auto, left or right")
    if vertebra_model not in (None, "", FULL_SPINE_MODEL):
        raise ValueError(f"Unknown full-spine vertebra model; available: {FULL_SPINE_MODEL}")
    if femoral_model or s1_model:
        raise ValueError("Separate femoral and S1 model overrides are not available for global SVA")


async def prediction_request(
    file: UploadFile = File(...), modality: str = Form(...), body_part: str = Form(...),
    view: str | None = Form(None), laterality: str | None = Form(None),
    vertebra_model: str | None = Form(None), femoral_model: str | None = Form(None),
    s1_model: str | None = Form(None), calibration: str | None = Form(None),
    processing_mode: str = Form("standard"), cpu_threads: int = Form(2),
    crop_localizer: bool = Form(True),
    crop_method: str = Form("search"),
    view_selection: str = Form("landmarks"),
    toolbar_removal: bool = Form(False),
    processor: str = Form("cpu"),
    anterior_side: str | None = Form(None),
):
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded file exceeds 50 MB")
    try:
        settings = runtime.parse_options(processing_mode, cpu_threads, crop_localizer,
                                         toolbar_removal, processor, crop_method, view_selection)
        if body_part.strip().lower() == "cervical":
            _validate_cervical_request(modality, view, laterality, vertebra_model,
                                      femoral_model, s1_model, anterior_side)
        elif body_part.strip().lower() == "full_spine":
            _validate_full_spine_request(modality, view, laterality, vertebra_model,
                                        femoral_model, s1_model, anterior_side)
        elif body_part.strip().lower() == "auto":
            _validate_auto_request(modality, view, laterality, vertebra_model,
                                   femoral_model, s1_model, anterior_side)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"settings": settings, "payload": payload, "modality": modality, "body_part": body_part,
            "view": view, "laterality": laterality, "vertebra_model": vertebra_model,
            "femoral_model": femoral_model, "s1_model": s1_model, "calibration": calibration,
            "anterior_side": anterior_side}


def run_prediction(request, reporter=None, cancelled=None):
    request = dict(request)
    settings = request.pop("settings")
    with runtime.session(settings, reporter, cancelled):
        if settings.low_memory:
            release_models()
        try:
            try:
                if runtime.processor().kind == "gpu" and runtime.processor().provider == processors.DIRECTML:
                    try:
                        from .gpu_parity import ensure_verified
                    except ImportError:  # uvicorn server:app from backend/
                        from gpu_parity import ensure_verified
                    ensure_verified(runtime.processor())
                return _analyze(**request)
            except runtime.GpuFailure as error:
                release_models()
                runtime.fallback_to_cpu(error)
                return _analyze(**request)
        finally:
            if settings.low_memory:
                release_models()


@app.post("/classify-view", summary="Preview the experimental whole-film view classifier")
async def classify_view(file: UploadFile = File(...)):
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=422, detail="Select an image smaller than 50 MB.")
    def classify():
        try:
            with runtime.session(runtime.parse_options(view_selection="classifier")):
                return detect_film(_decode_grayscale(payload))
        except (ValueError, OSError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    return await run_in_threadpool(classify)


@app.post("/decode", summary="Decode the original film for display windowing")
async def decode(file: UploadFile = File(...)) -> dict:
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=422, detail="Select an image smaller than 50 MB.")
    def decode_original():
        try:
            return decode_film(payload)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    return await run_in_threadpool(decode_original)


@app.post("/predict", summary="Find landmarks and measure a lateral spine radiograph")
async def predict(request=Depends(prediction_request)):
    return await run_in_threadpool(run_prediction, request)


@app.post("/predict-stream", summary="Process an image with live progress and heartbeats")
async def predict_stream(request=Depends(prediction_request)):
    return StreamingResponse(stream_job(lambda report, cancel: run_prediction(request, report, cancel)),
                             media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


def _analyze(payload, modality, body_part, view, laterality,
             vertebra_model, femoral_model, s1_model, calibration, anterior_side=None):
    body_part = body_part.strip().lower()
    detection = None
    requested_anterior_side = anterior_side
    try:
        runtime.report("decoding", "Reading the original image")
        pixel_array = _decode_grayscale(payload)
        if body_part == "auto":
            _validate_auto_request(modality, view, laterality, vertebra_model,
                                   femoral_model, s1_model, anterior_side)
            runtime.report("detecting", "Identifying the spine region")
            detection = detect_film(pixel_array, anterior_side=anterior_side)
            body_part = detection.get("body_part")
            if body_part not in ("cervical", "lumbar", "full_spine"):
                reasons = " ".join(detection.get("warnings", []))
                raise ValueError(reasons or "Could not confidently identify the spine region. Choose cervical, lumbar or full spine, then segment again.")
            if anterior_side not in ("left", "right") and body_part != "full_spine":
                anterior_side = detection.get("anterior_side")
        is_cervical = body_part == "cervical"
        is_full_spine = body_part == "full_spine"
        if is_cervical:
            _validate_cervical_request(modality, view, laterality, vertebra_model, femoral_model, s1_model, anterior_side)
        elif is_full_spine:
            _validate_full_spine_request(modality, view, laterality, vertebra_model, femoral_model, s1_model, anterior_side)
        elif body_part == "lumbar":
            from .models.models import _validate_supported_input
            _validate_supported_input(modality, body_part, view, laterality)
            if vertebra_model not in (None, "", "hrnet") or femoral_model not in (None, "", "unet") or s1_model not in (None, "", "keypointrcnn"):
                raise ValueError("The fixed lumbar pipeline uses HRNet, femoral U-Net and the S1 detector")
        else:
            raise ValueError("Unsupported spine region")
        from .fixed_pipeline import predict as predict_fixed
        evidence = detection.get("_orientation_evidence") if detection else None
        if is_cervical:
            prediction = cervical_prediction(pixel_array, anterior_side, model=CERVICAL_MODEL)
        elif is_full_spine:
            options = {"detection_evidence": evidence} if evidence else {}
            prediction = full_spine_prediction(pixel_array, anterior_side, model=FULL_SPINE_MODEL, **options)
        else:
            prediction = predict_fixed(pixel_array, anterior_side, region=body_part, detection_evidence=evidence)
        if runtime.options().low_memory:
            release_models()
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    runtime.report("encoding", "Preparing the image and segmentation overlays")
    encoded = {}
    for name in ("image", "mask", "femoral_mask"):
        output = io.BytesIO()
        Image.fromarray(prediction[name]).save(output, format="PNG", optimize=True)
        encoded[f"{name}_png"] = base64.b64encode(output.getvalue()).decode("ascii")
    # Every run, including the serial batch, reads the ORIGINAL image's ruler. The
    # inference crop can exclude it. Calibration failure must not lose segmentation.
    try:
        try:
            cached = json.loads(calibration) if calibration else None
        except (ValueError, TypeError):
            cached = None
        runtime.report("calibration", "Checking image calibration")
        image_calibration = calibration_from_payload(
            payload, include_preview=False, cached=cached,
        )
        image_calibration.pop('image_png', None)
    except runtime.Cancelled:
        raise
    except Exception:
        logging.getLogger(__name__).exception('Optional image calibration failed')
        image_calibration = {
            'version': 1, 'source_sha256': hashlib.sha256(payload).hexdigest(),
            'width': int(pixel_array.shape[1]), 'height': int(pixel_array.shape[0]),
            'coordinate_space': 'original_image', 'status': 'unavailable',
            'spacing': None, 'candidates': [], 'selected_index': None,
            'message': 'Automatic calibration unavailable. Review the reference in Image calibration.',
        }
    runtime.report("measuring", "Calculating global, cervical and lumbar measurements" if is_full_spine
                   else "Calculating regional spine measurements")
    spacing = image_calibration.get("spacing")
    # Calibration has already checked the digest/dimensions of the untouched
    # upload, and landmark models restore every point to that same frame.
    geometry = {**prediction["landmarks"], "region": body_part,
                "anterior_side": prediction["landmarks"].get("anterior_side") or anterior_side,
                "source_sha256": image_calibration["source_sha256"],
                "image_width": int(pixel_array.shape[1]), "image_height": int(pixel_array.shape[0]),
                "coordinate_space": "original_image",
                "pixel_spacing": [spacing["row_mm"], spacing["column_mm"]] if spacing else None,
                "spacing_source": spacing.get("source") if spacing else None}
    try:
        analysis = global_sva_measurements_from_geometry(geometry)
    except (AttributeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    # Store model, crop, scale and resource provenance alongside the measurements.
    qc = {**prediction.get("qc", {}), **analysis.get("qc", {}),
          "models": prediction["models"], "model_versions": model_versions(prediction["models"]),
          "framing": prediction["framing"],
          "processing": {"view_selection": runtime.options().view_selection,
                         "mode": runtime.options().mode,
                         "cpu_threads": runtime.options().inference_threads,
                         "runtime": "onnxruntime", "runtime_version": ort.__version__,
                         "providers": runtime.providers(),
                         "batches": runtime.batches(),
                         "processor": runtime.processor_record(),
                         "crop_localizer": not is_cervical,
                         "crop_method": "not_applicable" if is_cervical else runtime.options().crop_method,
                         "toolbar_removal": False,
                         "search_batch": runtime.options().search_batch}}
    # Landmark pipelines return the original
    # image. Retain user preferences separately from
    # the processing actually applied by this model.
    qc["processing"]["requested_crop_localizer"] = runtime.options().crop_localizer
    if is_cervical:
        qc["processing"]["requested_crop_method"] = runtime.options().crop_method
    qc["processing"]["requested_toolbar_removal"] = runtime.options().toolbar_removal
    for field in ("provenance", "warnings"):
        if field in prediction:
            qc[field] = prediction[field]
    if is_full_spine and prediction["framing"].get("femoral") is not None:
        qc["femoral"] = prediction["framing"]["femoral"]
    if detection is not None:
        qc["film_detection"] = {key: value for key, value in detection.items() if not key.startswith("_")}
        # Existing lumbar measurements omit their region; Auto results must carry
        # the resolved type so saving, display and export do not use the preference.
        analysis["geometry"]["region"] = body_part
        analysis["measurements"]["region"] = body_part
    runtime.report("complete", "Measurements ready")
    # No prediction places T1 or T4; the keys are present, null, until a person places them.
    analysis = _with_thoracic({}, analysis)
    return {**encoded, **analysis, "qc": qc, "labels": {},
            "calibration": image_calibration}


def _with_thoracic(request: dict, result: dict) -> dict:
    """Add T1S, T4PA and the two echoed points to a measured result, in place."""
    thoracic = thoracic_measurements(request, result["geometry"])
    result["measurements"].update(thoracic["measurements"])
    result["geometry"].update(thoracic["geometry"])
    return result


@app.post("/measure", summary="Recalculate measurements from corrected landmarks")
async def measure(geometry: dict[str, object]) -> dict[str, object]:
    """Return measurements after interactive landmark correction."""

    try:
        if geometry.get("region") == "cervical":
            result = await run_in_threadpool(cervical_measurements_from_geometry, geometry)
        elif geometry.get("region") == "full_spine" or geometry.get("pipeline_revision"):
            result = await run_in_threadpool(global_sva_measurements_from_geometry, geometry)
        elif geometry.get("region") not in (None, "lumbar"):
            raise ValueError("Unknown geometry region; available: lumbar, cervical, full_spine")
        else:
            result = await run_in_threadpool(
                spinopelvic_measurements_from_geometry,
                geometry.get("vertebrae"),
                geometry.get("s1_superior"),
                geometry.get("femoral_circles"),
                allow_empty=True,
            )
            if geometry.get("region") == "lumbar":
                result["geometry"]["region"] = "lumbar"
                result["measurements"]["region"] = "lumbar"
        return _with_thoracic(geometry, result)
    except (AttributeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/models", summary="Which model can read which structure")
def models(body_part: str = "lumbar") -> dict[str, list[str]]:
    if body_part.strip().lower() == "auto":
        return {"vertebrae": [], "femoral": [], "s1": []}
    if body_part.strip().lower() == "cervical":
        return {"vertebrae": [CERVICAL_MODEL], "femoral": [], "s1": []}
    if body_part.strip().lower() == "full_spine":
        return {"vertebrae": [FULL_SPINE_MODEL], "femoral": [], "s1": []}
    if body_part.strip().lower() != "lumbar":
        raise HTTPException(status_code=422, detail="Unknown body part; available: lumbar, cervical, full_spine")
    return {"vertebrae": ["hrnet"], "femoral": ["unet"], "s1": ["keypointrcnn"]}


@app.get("/processors", summary="The CPU and each GPU the bundled runtime can run the models on")
def list_processors() -> dict[str, list[dict[str, object]]]:
    return {"processors": [processor.public() for processor in processors.available()]}


@app.get("/health", include_in_schema=False)
def health() -> dict[str, str]:
    return {"status": "ok"}


async def calibration_request(file: UploadFile = File(...), profile: str | None = Form(None),
                              calibration: str | None = Form(None),
                              include_preview: bool = Form(True), preview_only: bool = Form(False),
                              processing_mode: str = Form("standard"), cpu_threads: int = Form(2)):
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="The selected file is empty")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The selected file exceeds 50 MB")
    try:
        profile = validate_profile(json.loads(profile)) if profile else None
        cached = json.loads(calibration) if calibration else None
        policy = runtime.parse_options(processing_mode, cpu_threads)
    except Exception as error:
        raise HTTPException(status_code=422, detail="Invalid calibration settings") from error
    return {"payload": payload, "profile": profile, "include_preview": include_preview,
            "preview_only": preview_only, "policy": policy, "cached": cached}


def run_calibration(request, reporter=None, cancelled=None):
    with runtime.session(request['policy'], reporter, cancelled):
        if request['policy'].low_memory:
            release_models()
        try:
            runtime.report("calibration", "Checking image calibration")
            result = calibration_from_payload(request['payload'], request['profile'],
                                              request['include_preview'], request['preview_only'], cached=request['cached'])
            runtime.checkpoint()
            return result
        except runtime.Cancelled:
            raise
        except Exception as error:
            raise HTTPException(status_code=422, detail="Could not read the image for calibration") from error
        finally:
            if request['policy'].low_memory:
                release_models()


@app.post("/calibrate", summary="Read image scale from an original radiograph")
async def calibrate(request=Depends(calibration_request)):
    return await run_in_threadpool(run_calibration, request)


@app.post("/calibrate-stream", summary="Read image scale with progress and heartbeats")
async def calibrate_stream(request=Depends(calibration_request)):
    return StreamingResponse(stream_job(lambda report, cancel: run_calibration(request, report, cancel)),
                             media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.post("/calibration-profile", summary="Learn annotation color from a corrected reference")
async def calibration_profile(file: UploadFile = File(...), endpoints: str = Form(...)):
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Select an image smaller than 50 MB")
    try:
        return await run_in_threadpool(learn_profile, payload, json.loads(endpoints))
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
