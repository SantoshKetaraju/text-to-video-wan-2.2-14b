FROM runpod/worker-comfyui:5.8.6-base-cuda12.8.1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# =========================
# BASE RUNTIME
# =========================
# Stay on the same pinned RunPod ComfyUI image as the proven Wan character-
# replacement worker. This graph uses only native ComfyUI nodes.
RUN echo "=== BASE IMAGE CHECK ===" && \
    python --version && \
    pip --version && \
    which comfy && \
    test -d /comfyui

# SaveVideo produces the source MP4. FFmpeg optionally applies the customer
# disclosure watermark immediately before upload.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* && \
    ffmpeg -version >/dev/null && \
    ffmpeg -hide_banner -filters 2>/dev/null | grep drawtext && \
    test -f /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf

# =========================
# RUNPOD CACHED MODEL CONTRACT
# =========================
# Configure the RunPod endpoint Model field with this private repository:
#   https://huggingface.co/PolarApparel/ethrylsynth-wan-models
#
# RunPod mounts the active Hugging Face snapshot before the worker starts.
# handler.py will validate the required files and symlink them into the normal
# ComfyUI model directories without copying or downloading them during a job.
ENV WAN_T2V_CACHE_REPO=PolarApparel/ethrylsynth-wan-models
ENV WAN_T2V_CACHE_ROOT=/runpod-volume/huggingface-cache/hub/models--PolarApparel--ethrylsynth-wan-models

# Required weights must come from the RunPod-mounted cache. Prevent libraries
# from silently downloading substitutes during billed worker execution.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

RUN mkdir -p \
    /comfyui/models/diffusion_models \
    /comfyui/models/loras \
    /comfyui/models/text_encoders \
    /comfyui/models/vae

# =========================
# WORKER CODE AND WORKFLOW
# =========================
COPY workflow.json /workflow.json
COPY handler.py /handler.py
COPY watermark.py /watermark.py

ENV RUNPOD_HANDLER=/handler.py
ENV WAN_T2V_WORKFLOW_PATH=/workflow.json

# Catch Python syntax errors and accidental graph/model drift during the image
# build. The two four-step LoRAs are cached for a future fast graph but are not
# required by this full-quality workflow.
RUN python -m compileall -q /handler.py /watermark.py && \
    python - <<'PY'
import json
from pathlib import Path


workflow = json.loads(Path("/workflow.json").read_text(encoding="utf-8"))

expected_model_names = {
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "wan_2.1_vae.safetensors",
    "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
}
serialized = json.dumps(workflow)
missing_models = sorted(
    name for name in expected_model_names if name not in serialized
)
assert not missing_models, (
    f"Workflow does not reference cached models: {missing_models}"
)

native_node_classes = {
    "CLIPLoader",
    "CLIPTextEncode",
    "CreateVideo",
    "EmptyHunyuanLatentVideo",
    "KSamplerAdvanced",
    "ModelSamplingSD3",
    "SaveVideo",
    "UNETLoader",
    "VAEDecode",
    "VAELoader",
}
workflow_node_classes = {
    node["class_type"] for node in workflow.values()
}
unexpected_nodes = sorted(workflow_node_classes - native_node_classes)
assert not unexpected_nodes, (
    f"Workflow unexpectedly requires additional nodes: {unexpected_nodes}"
)

assert workflow["104"]["inputs"]["length"] % 4 == 1, (
    "Wan video frame count must use the 4n+1 form"
)
print("Workflow structure and cached-model names verified")
PY

RUN echo "=== WAN 2.2 TEXT-TO-VIDEO BUILD COMPLETE ==="
