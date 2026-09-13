from __future__ import annotations

import copy
import json
import math
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import requests
import runpod

from watermark import apply_disclosure_watermark


# ============================================================
# FIXED WAN 2.2 TEXT-TO-VIDEO PIPELINE
# ============================================================

TOOL_ID = "text2media"
MODEL_ID = "wan_2_2_t2v_14b"
WORKFLOW_PATH = Path(
    os.environ.get("WAN_T2V_WORKFLOW_PATH", "/workflow.json")
)

MAX_PROMPT_LENGTH = 2000
MAX_TOTAL_SECONDS = int(os.environ.get("MAX_TOTAL_SECONDS", "3600"))
WORKFLOW_TIMEOUT_SECONDS = int(
    os.environ.get("WORKFLOW_TIMEOUT_SECONDS", "3300")
)

COMFY_BASE_URL = os.environ.get(
    "COMFY_BASE_URL", "http://127.0.0.1:8188"
)
COMFY_OUTPUT = Path(os.environ.get("COMFY_OUTPUT", "/comfyui/output"))
WORKSPACE_ROOT = Path(
    os.environ.get("JOB_WORKSPACE_ROOT", "/tmp/ethrylsynth")
)

POSITIVE_PROMPT_NODE_ID = "99"
NEGATIVE_PROMPT_NODE_ID = "91"
HIGH_NOISE_SHIFT_NODE_ID = "93"
LOW_NOISE_SHIFT_NODE_ID = "94"
LOW_NOISE_SAMPLER_NODE_ID = "95"
HIGH_NOISE_SAMPLER_NODE_ID = "96"
SAVE_VIDEO_NODE_ID = "98"
CREATE_VIDEO_NODE_ID = "100"
LATENT_VIDEO_NODE_ID = "104"

WORKFLOW_DEFAULTS = {
    "positive_prompt": (
        "Beautiful young European woman with honey blonde hair gracefully "
        "turning her head back over shoulder, gentle smile, bright eyes "
        "looking at camera. Hair flowing in slow motion as she turns. Soft "
        "natural lighting, clean background, cinematic portrait."
    ),
    "negative_prompt": (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，"
        "画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
        "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
        "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
        "三条腿，背景人很多，倒着走"
    ),
    "quality": "standard",
    "aspect_ratio": "landscape",
    "duration_seconds": 5,
    "frame_rate": 16,
    "seed": 0,
    "generation_mode": "quality",
    "steps": 20,
    "cfg": 3.5,
    "phase_switch_step": 10,
    "high_noise_shift": 8.0,
    "low_noise_shift": 8.0,
}

RESOLUTIONS = {
    "standard": {
        "landscape": (832, 480),
        "portrait": (480, 832),
    },
    "hd": {
        "landscape": (1280, 720),
        "portrait": (720, 1280),
    },
}

# status.json is a presentation overlay only. Keep messages non-technical and
# never expose model names, graph structure, node numbers, prompts, or errors.
STATUS_MESSAGES = {
    "started": "GPU acquired. Preparing your video...",
    "generating": "Creating your video...",
    "upload": "Finalizing your video...",
    "done": "Video completed successfully",
    "failed": "Video generation failed",
    "timeout": "Video generation took too long",
}


# ============================================================
# RUNPOD MODEL CACHE
# ============================================================

WAN_T2V_CACHE_ROOT = Path(
    os.environ.get(
        "WAN_T2V_CACHE_ROOT",
        (
            "/runpod-volume/huggingface-cache/hub/"
            "models--PolarApparel--ethrylsynth-wan-2.2-t2v"
        ),
    )
)

REQUIRED_CACHED_MODEL_LINKS = {
    Path(
        "diffusion_models/"
        "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    ): Path(
        "/comfyui/models/diffusion_models/"
        "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    ),
    Path(
        "diffusion_models/"
        "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
    ): Path(
        "/comfyui/models/diffusion_models/"
        "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
    ),
    Path("text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"): Path(
        "/comfyui/models/text_encoders/"
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    ),
    Path("vae/wan_2.1_vae.safetensors"): Path(
        "/comfyui/models/vae/wan_2.1_vae.safetensors"
    ),
}

# These files are present in the cache for a future separately tested fast
# graph. Their absence must not prevent the current quality graph from booting.
OPTIONAL_CACHED_MODEL_LINKS = {
    Path(
        "loras/"
        "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
    ): Path(
        "/comfyui/models/loras/"
        "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
    ),
    Path(
        "loras/"
        "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"
    ): Path(
        "/comfyui/models/loras/"
        "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"
    ),
}

MINIMUM_MODEL_BYTES = {
    "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors": 10_000_000_000,
    "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors": 10_000_000_000,
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors": 6_000_000_000,
    "wan_2.1_vae.safetensors": 200_000_000,
    "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors": (
        1_000_000_000
    ),
    "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors": (
        1_000_000_000
    ),
}


# ============================================================
# LOGGING AND FAILURES
# ============================================================

def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[{timestamp}] {message}", flush=True)


def fail_job(reason: str) -> None:
    log(f"[FAIL] {reason}")
    raise RuntimeError(reason)


# ============================================================
# MODEL AND WORKFLOW ASSETS
# ============================================================

def _cached_snapshot_candidates() -> list[Path]:
    snapshots_directory = WAN_T2V_CACHE_ROOT / "snapshots"
    candidates: list[Path] = []
    main_ref = WAN_T2V_CACHE_ROOT / "refs" / "main"

    if main_ref.is_file():
        revision = main_ref.read_text(
            encoding="utf-8", errors="ignore"
        ).strip()
        if revision:
            candidates.append(snapshots_directory / revision)

    if snapshots_directory.is_dir():
        candidates.extend(
            sorted(
                (
                    path
                    for path in snapshots_directory.iterdir()
                    if path.is_dir()
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )

    unique: dict[str, Path] = {}
    for candidate in candidates:
        unique[str(candidate)] = candidate
    return list(unique.values())


def _resolve_cached_snapshot() -> Path:
    required_paths = tuple(REQUIRED_CACHED_MODEL_LINKS)

    for _ in range(15):
        for snapshot in _cached_snapshot_candidates():
            if all((snapshot / path).is_file() for path in required_paths):
                log(f"Resolved cached Wan T2V snapshot: {snapshot}")
                return snapshot.resolve()
        time.sleep(1)

    fail_job(
        "RunPod cached model snapshot is missing or incomplete. Configure "
        "the endpoint Model field with https://huggingface.co/PolarApparel/"
        "ethrylsynth-wan-2.2-t2v"
    )


def _link_model(source: Path, destination: Path, required: bool) -> None:
    if not source.is_file():
        if required:
            fail_job(f"Required cached model is missing: {source}")
        log(f"Optional cached model is not present: {source.name}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        try:
            if destination.resolve() == source.resolve():
                log(f"Cached model link already present: {destination}")
                return
        except OSError:
            pass
        destination.unlink()
    elif destination.exists():
        fail_job(
            "Refusing to replace unexpected model file in container: "
            f"{destination}"
        )

    destination.symlink_to(source.resolve())
    log(f"Linked cached model: {destination} -> {source}")


def link_cached_models() -> None:
    snapshot = _resolve_cached_snapshot()

    for relative_source, destination in REQUIRED_CACHED_MODEL_LINKS.items():
        _link_model(snapshot / relative_source, destination, required=True)
    for relative_source, destination in OPTIONAL_CACHED_MODEL_LINKS.items():
        _link_model(snapshot / relative_source, destination, required=False)


def _validate_model(path: Path) -> None:
    if not path.is_file():
        fail_job(f"Required model is missing: {path}")
    size = path.stat().st_size
    minimum = MINIMUM_MODEL_BYTES[path.name]
    if size < minimum:
        fail_job(
            f"Required model is unexpectedly small: {path} ({size} bytes)"
        )
    log(f"Verified model: {path.name} ({size / (1024 ** 3):.2f} GiB)")


def validate_runtime_assets() -> None:
    if not WORKFLOW_PATH.is_file():
        fail_job(f"Workflow is missing: {WORKFLOW_PATH}")
    for model_path in REQUIRED_CACHED_MODEL_LINKS.values():
        _validate_model(model_path)


# ============================================================
# R2 STATUS AND OUTPUT
# ============================================================

def _r2_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _status_key(app_job_id: str, user_id: str) -> str:
    return f"tmp/{user_id}/{TOOL_ID}/job_{app_job_id}/status.json"


def write_status(
    app_job_id: str,
    user_id: str,
    phase: str,
    *,
    state: str = "processing",
) -> None:
    message = STATUS_MESSAGES[phase]
    payload = {
        "state": state,
        "phase": phase,
        "message": message,
        "ts": time.time(),
    }
    log(f"[{app_job_id}] STATUS -> {payload}")

    local_path = Path(f"/tmp/status_{app_job_id}.json")
    local_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        _r2_client().upload_file(
            str(local_path),
            os.environ["R2_BUCKET"],
            _status_key(app_job_id, user_id),
            ExtraArgs={
                "ContentType": "application/json",
                "CacheControl": "no-store",
            },
        )
    except Exception as exc:
        # Status is UX-only and must never terminate a valid generation.
        log(f"[{app_job_id}] Status upload failed: {exc}")


def upload_video(local_path: Path, object_key: str) -> None:
    if not local_path.is_file() or local_path.stat().st_size <= 0:
        fail_job("Generated video is missing or empty")
    _r2_client().upload_file(
        str(local_path),
        os.environ["R2_BUCKET"],
        object_key,
        ExtraArgs={
            "ContentType": "video/mp4",
            "CacheControl": "public, max-age=3600",
        },
    )


# ============================================================
# INPUT CONTRACT
# ============================================================

def _required_string(data: dict[str, Any], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        fail_job(f"Missing required input: {name}")
    return value.strip()


def _valid_uuid(value: str, name: str) -> str:
    try:
        return str(uuid.UUID(value))
    except ValueError:
        fail_job(f"Invalid {name}")


def _text_or_default(
    value: object,
    *,
    field_name: str,
    default: str,
    allow_empty: bool,
) -> str:
    if not isinstance(value, str):
        if value is not None:
            log(f"[INPUT] Invalid {field_name}; using workflow default")
        return default
    normalized = value.strip()
    if len(normalized) > MAX_PROMPT_LENGTH or (not normalized and not allow_empty):
        log(f"[INPUT] Invalid {field_name}; using workflow default")
        return default
    return normalized


def _integer_or_default(
    value: object,
    *,
    field_name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or value > maximum
    ):
        if value is not None:
            log(f"[INPUT] Invalid {field_name}; using workflow default")
        return default
    return value


def _number_or_default(
    value: object,
    *,
    field_name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < minimum
        or value > maximum
    ):
        if value is not None:
            log(f"[INPUT] Invalid {field_name}; using workflow default")
        return default
    return float(value)


def _choice_or_default(
    value: object,
    *,
    field_name: str,
    default: str,
    choices: set[str],
) -> str:
    if value not in choices:
        if value is not None:
            log(f"[INPUT] Invalid {field_name}; using workflow default")
        return default
    return str(value)


def normalize_input(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("input") or {}
    if not isinstance(data, dict):
        fail_job("RunPod input must be an object")

    app_job_id = _valid_uuid(_required_string(data, "jobId"), "jobId")
    user_id = _valid_uuid(_required_string(data, "userId"), "userId")
    output_path = _required_string(data, "outputPath")
    model = _required_string(data, "model")

    if model != MODEL_ID:
        fail_job(f"Invalid model: {model}")
    expected_output_path = (
        f"tmp/{user_id}/{TOOL_ID}/job_{app_job_id}/output/output.mp4"
    )
    if output_path != expected_output_path:
        fail_job("Invalid outputPath")

    # Only the literal JSON boolean true enables watermarking. Null, false,
    # missing, strings, and numbers all leave the generated video untouched.
    watermark = data.get("watermark") is True

    positive_prompt = _text_or_default(
        data.get("positive_prompt"),
        field_name="positive_prompt",
        default=WORKFLOW_DEFAULTS["positive_prompt"],
        allow_empty=False,
    )
    negative_prompt = _text_or_default(
        data.get("negative_prompt"),
        field_name="negative_prompt",
        default=WORKFLOW_DEFAULTS["negative_prompt"],
        allow_empty=True,
    )
    quality = _choice_or_default(
        data.get("quality"),
        field_name="quality",
        default=WORKFLOW_DEFAULTS["quality"],
        choices=set(RESOLUTIONS),
    )
    aspect_ratio = _choice_or_default(
        data.get("aspect_ratio"),
        field_name="aspect_ratio",
        default=WORKFLOW_DEFAULTS["aspect_ratio"],
        choices={"landscape", "portrait"},
    )
    duration_seconds = _integer_or_default(
        data.get("duration_seconds"),
        field_name="duration_seconds",
        default=WORKFLOW_DEFAULTS["duration_seconds"],
        minimum=1,
        maximum=10,
    )
    frame_rate = _integer_or_default(
        data.get("frame_rate"),
        field_name="frame_rate",
        default=WORKFLOW_DEFAULTS["frame_rate"],
        minimum=16,
        maximum=16,
    )
    seed = _integer_or_default(
        data.get("seed"),
        field_name="seed",
        default=WORKFLOW_DEFAULTS["seed"],
        minimum=0,
        maximum=4_294_967_295,
    )

    generation_mode = data.get(
        "generation_mode", WORKFLOW_DEFAULTS["generation_mode"]
    )
    if generation_mode != "quality":
        fail_job(
            "The current workflow supports quality generation only; "
            "the four-step LoRA graph has not been enabled"
        )

    steps = _integer_or_default(
        data.get("steps"),
        field_name="steps",
        default=WORKFLOW_DEFAULTS["steps"],
        minimum=4,
        maximum=40,
    )
    cfg = _number_or_default(
        data.get("cfg"),
        field_name="cfg",
        default=WORKFLOW_DEFAULTS["cfg"],
        minimum=1.0,
        maximum=8.0,
    )
    default_phase_switch = min(
        WORKFLOW_DEFAULTS["phase_switch_step"], steps - 1
    )
    phase_switch_step = _integer_or_default(
        data.get("phase_switch_step"),
        field_name="phase_switch_step",
        default=default_phase_switch,
        minimum=1,
        maximum=steps - 1,
    )
    high_noise_shift = _number_or_default(
        data.get("high_noise_shift"),
        field_name="high_noise_shift",
        default=WORKFLOW_DEFAULTS["high_noise_shift"],
        minimum=1.0,
        maximum=16.0,
    )
    low_noise_shift = _number_or_default(
        data.get("low_noise_shift"),
        field_name="low_noise_shift",
        default=WORKFLOW_DEFAULTS["low_noise_shift"],
        minimum=1.0,
        maximum=16.0,
    )

    # Width, height and frame count are always derived from validated settings.
    # Browser- or direct-API-supplied values can never multiply GPU cost.
    width, height = RESOLUTIONS[quality][aspect_ratio]
    frame_count = duration_seconds * frame_rate + 1
    if frame_count % 4 != 1:
        fail_job("Derived frame count must use the Wan 4n+1 form")

    for field_name, supplied, derived in (
        ("width", data.get("width"), width),
        ("height", data.get("height"), height),
        ("frame_count", data.get("frame_count"), frame_count),
    ):
        if supplied is not None and supplied != derived:
            log(f"[INPUT] Ignoring non-canonical {field_name}: {supplied}")

    return {
        "app_job_id": app_job_id,
        "user_id": user_id,
        "output_path": output_path,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "quality": quality,
        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
        "frame_rate": frame_rate,
        "frame_count": frame_count,
        "seed": seed,
        "generation_mode": generation_mode,
        "steps": steps,
        "cfg": cfg,
        "phase_switch_step": phase_switch_step,
        "high_noise_shift": high_noise_shift,
        "low_noise_shift": low_noise_shift,
        "watermark": watermark,
    }


# ============================================================
# COMFYUI WORKFLOW
# ============================================================

def wait_for_comfyui(timeout_seconds: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = requests.get(
                f"{COMFY_BASE_URL}/system_stats", timeout=2
            )
            if response.ok:
                log("ComfyUI is ready")
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    fail_job("ComfyUI did not become ready")


def load_workflow_template() -> dict[str, Any]:
    try:
        workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail_job(f"Could not read workflow: {exc}")
    if not isinstance(workflow, dict):
        fail_job("Workflow must contain a JSON object")

    required_nodes = {
        POSITIVE_PROMPT_NODE_ID: "CLIPTextEncode",
        NEGATIVE_PROMPT_NODE_ID: "CLIPTextEncode",
        HIGH_NOISE_SHIFT_NODE_ID: "ModelSamplingSD3",
        LOW_NOISE_SHIFT_NODE_ID: "ModelSamplingSD3",
        LOW_NOISE_SAMPLER_NODE_ID: "KSamplerAdvanced",
        HIGH_NOISE_SAMPLER_NODE_ID: "KSamplerAdvanced",
        SAVE_VIDEO_NODE_ID: "SaveVideo",
        CREATE_VIDEO_NODE_ID: "CreateVideo",
        LATENT_VIDEO_NODE_ID: "EmptyHunyuanLatentVideo",
        "90": "CLIPLoader",
        "92": "VAELoader",
        "97": "VAEDecode",
        "101": "UNETLoader",
        "102": "UNETLoader",
    }
    for node_id, expected_class in required_nodes.items():
        node = workflow.get(node_id)
        if not isinstance(node, dict) or node.get("class_type") != expected_class:
            fail_job(f"Workflow node {node_id} must be {expected_class}")
    return workflow


def build_workflow(
    template: dict[str, Any], job: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    workflow = copy.deepcopy(template)
    output_prefix = f"text_to_video_{job['app_job_id']}"

    workflow[POSITIVE_PROMPT_NODE_ID]["inputs"]["text"] = (
        job["positive_prompt"]
    )
    workflow[NEGATIVE_PROMPT_NODE_ID]["inputs"]["text"] = (
        job["negative_prompt"]
    )
    workflow[HIGH_NOISE_SHIFT_NODE_ID]["inputs"]["shift"] = (
        job["high_noise_shift"]
    )
    workflow[LOW_NOISE_SHIFT_NODE_ID]["inputs"]["shift"] = (
        job["low_noise_shift"]
    )

    for node_id in (HIGH_NOISE_SAMPLER_NODE_ID, LOW_NOISE_SAMPLER_NODE_ID):
        workflow[node_id]["inputs"]["steps"] = job["steps"]
        workflow[node_id]["inputs"]["cfg"] = job["cfg"]
        workflow[node_id]["inputs"]["noise_seed"] = job["seed"]

    workflow[HIGH_NOISE_SAMPLER_NODE_ID]["inputs"]["start_at_step"] = 0
    workflow[HIGH_NOISE_SAMPLER_NODE_ID]["inputs"]["end_at_step"] = (
        job["phase_switch_step"]
    )
    workflow[LOW_NOISE_SAMPLER_NODE_ID]["inputs"]["start_at_step"] = (
        job["phase_switch_step"]
    )
    workflow[LOW_NOISE_SAMPLER_NODE_ID]["inputs"]["end_at_step"] = 10000

    latent_inputs = workflow[LATENT_VIDEO_NODE_ID]["inputs"]
    latent_inputs["width"] = job["width"]
    latent_inputs["height"] = job["height"]
    latent_inputs["length"] = job["frame_count"]
    latent_inputs["batch_size"] = 1
    workflow[CREATE_VIDEO_NODE_ID]["inputs"]["fps"] = job["frame_rate"]
    workflow[SAVE_VIDEO_NODE_ID]["inputs"]["filename_prefix"] = output_prefix
    return workflow, output_prefix


def _submit_workflow(workflow: dict[str, Any], app_job_id: str) -> str:
    response = requests.post(
        f"{COMFY_BASE_URL}/prompt",
        json={"prompt": workflow, "client_id": app_job_id},
        timeout=60,
    )
    if not response.ok:
        fail_job(f"ComfyUI rejected the workflow: {response.text[-2000:]}")
    prompt_id = response.json().get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        fail_job("ComfyUI did not return a prompt ID")
    return prompt_id


def _history_error(history: dict[str, Any]) -> str:
    messages = history.get("status", {}).get("messages") or []
    for message in reversed(messages):
        if not isinstance(message, list) or len(message) < 2:
            continue
        if message[0] != "execution_error" or not isinstance(message[1], dict):
            continue
        detail = message[1].get("exception_message")
        node_type = message[1].get("node_type")
        if detail:
            return f"{node_type or 'workflow node'}: {detail}"
    return "ComfyUI reported a workflow execution error"


def _wait_for_workflow(
    prompt_id: str, global_deadline: float
) -> dict[str, Any]:
    workflow_deadline = min(
        time.monotonic() + WORKFLOW_TIMEOUT_SECONDS,
        global_deadline,
    )
    while time.monotonic() < workflow_deadline:
        try:
            response = requests.get(
                f"{COMFY_BASE_URL}/history/{prompt_id}", timeout=10
            )
            if response.ok:
                history = response.json().get(prompt_id)
                if isinstance(history, dict):
                    status = history.get("status") or {}
                    if status.get("status_str") == "error":
                        fail_job(_history_error(history))
                    if status.get("completed"):
                        return history
        except RuntimeError:
            raise
        except (requests.RequestException, ValueError):
            # Transient localhost polling failures must not destroy a job.
            pass
        time.sleep(1)

    try:
        requests.post(f"{COMFY_BASE_URL}/interrupt", timeout=10)
    except requests.RequestException:
        pass
    raise TimeoutError("Wan text-to-video workflow timed out")


def _safe_output_candidate(item: object) -> Path | None:
    if not isinstance(item, dict):
        return None
    filename = item.get("filename")
    if not isinstance(filename, str) or not filename:
        return None
    candidate = COMFY_OUTPUT
    subfolder = item.get("subfolder")
    if isinstance(subfolder, str) and subfolder:
        candidate /= subfolder
    candidate = (candidate / filename).resolve()
    try:
        candidate.relative_to(COMFY_OUTPUT.resolve())
    except ValueError:
        return None
    if candidate.suffix.lower() != ".mp4":
        return None
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        return None
    return candidate


def _find_output_video(
    history: dict[str, Any], output_prefix: str
) -> Path:
    candidates: dict[str, Path] = {}
    node_output = (
        (history.get("outputs") or {}).get(SAVE_VIDEO_NODE_ID) or {}
    )
    for key in ("videos", "gifs", "images"):
        for item in node_output.get(key) or []:
            candidate = _safe_output_candidate(item)
            if candidate is not None:
                candidates[str(candidate)] = candidate

    for candidate in COMFY_OUTPUT.rglob(f"{output_prefix}*.mp4"):
        resolved = candidate.resolve()
        if resolved.is_file() and resolved.stat().st_size > 0:
            candidates[str(resolved)] = resolved

    if not candidates:
        fail_job("ComfyUI produced no MP4 output")
    return max(candidates.values(), key=lambda path: path.stat().st_mtime)


def run_workflow(job: dict[str, Any], destination: Path) -> None:
    template = load_workflow_template()
    workflow, output_prefix = build_workflow(template, job)
    COMFY_OUTPUT.mkdir(parents=True, exist_ok=True)
    for stale_output in COMFY_OUTPUT.rglob(f"{output_prefix}*"):
        if stale_output.is_file():
            stale_output.unlink(missing_ok=True)

    prompt_id = _submit_workflow(workflow, job["app_job_id"])
    history = _wait_for_workflow(
        prompt_id,
        time.monotonic() + MAX_TOTAL_SECONDS,
    )
    generated = _find_output_video(history, output_prefix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.move(str(generated), str(destination))


# ============================================================
# BACKEND LIFECYCLE CALLBACKS
# ============================================================

def _post_with_retry(
    url: str,
    payload: dict[str, Any],
    app_job_id: str,
    label: str,
    retries: int = 3,
) -> bool:
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=10)
            log(
                f"[{app_job_id}] {label} attempt {attempt}: "
                f"{response.status_code} {response.text}"
            )
            if response.ok and response.json().get("ok") is True:
                return True
        except (requests.RequestException, ValueError, AttributeError) as exc:
            log(f"[{app_job_id}] {label} attempt {attempt} failed: {exc}")
        if attempt < retries:
            time.sleep(2)
    return False


def notify_job_started(app_job_id: str) -> None:
    url = os.environ.get("JOB_STARTED_BACKEND_URL")
    if url:
        _post_with_retry(
            url,
            {"jobId": app_job_id, "startedAt": int(time.time() * 1000)},
            app_job_id,
            "job-started",
        )


def notify_compute_ended(
    app_job_id: str,
    metadata: dict[str, Any],
    compute_ended_at: int,
) -> bool:
    url = os.environ.get("JOB_COMPUTE_ENDED_URL")
    if not url:
        return False
    return _post_with_retry(
        url,
        {
            "jobId": app_job_id,
            "computeEndedAt": compute_ended_at,
            "metadata": metadata,
        },
        app_job_id,
        "compute-ended",
    )


# ============================================================
# HANDLER
# ============================================================

def handler(event: dict[str, Any]) -> dict[str, Any]:
    log("WAN 2.2 TEXT-TO-VIDEO HANDLER: v1")
    log(f"RunPod job id: {event.get('id')}")

    validate_runtime_assets()
    job = normalize_input(event)
    app_job_id = job["app_job_id"]
    user_id = job["user_id"]
    workspace = WORKSPACE_ROOT / app_job_id
    output_path = workspace / "output.mp4"

    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)

    # This subset is accepted by the current shared callback validator. It is
    # enough to compare compute time by generated duration and resolution.
    metadata = {
        "schema_version": 1,
        "processing_fps": job["frame_rate"],
        "processing_width": job["width"],
        "processing_height": job["height"],
        "normalized_duration_seconds": job["duration_seconds"],
        "normalized_frame_count": job["frame_count"],
        "seed": job["seed"],
    }

    compute_ended_at: int | None = None
    compute_ended_sent = False
    notify_job_started(app_job_id)
    write_status(app_job_id, user_id, "started")

    try:
        wait_for_comfyui()
        write_status(app_job_id, user_id, "generating")
        run_workflow(job, output_path)

        compute_ended_at = int(time.time() * 1000)
        compute_ended_sent = notify_compute_ended(
            app_job_id, metadata, compute_ended_at
        )

        write_status(app_job_id, user_id, "upload")
        if job["watermark"]:
            log(f"[{app_job_id}] Applying disclosure watermark")
            apply_disclosure_watermark(output_path)
        upload_video(output_path, job["output_path"])
        write_status(app_job_id, user_id, "done", state="done")
        log(f"[{app_job_id}] SUCCESS")
        return {"ok": True}
    except TimeoutError as exc:
        write_status(app_job_id, user_id, "timeout", state="timeout")
        fail_job(str(exc))
    except Exception as exc:
        write_status(app_job_id, user_id, "failed", state="error")
        fail_job(str(exc))
    finally:
        if compute_ended_at is None:
            compute_ended_at = int(time.time() * 1000)
        if not compute_ended_sent:
            notify_compute_ended(app_job_id, metadata, compute_ended_at)
        shutil.rmtree(workspace, ignore_errors=True)


# ============================================================
# RUNPOD BOOTSTRAP
# ============================================================

def bootstrap() -> None:
    link_cached_models()
    validate_runtime_assets()
    runpod.serverless.start({"handler": handler})


if os.environ.get("WAN_T2V_SKIP_BOOTSTRAP") != "1":
    bootstrap()
