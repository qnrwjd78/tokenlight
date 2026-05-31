FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    ninja-build \
    pkg-config \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/requirements-runtime.txt

COPY docker/entrypoint.sh /usr/local/bin/tokenlight-entrypoint
RUN chmod +x /usr/local/bin/tokenlight-entrypoint

ENTRYPOINT ["/usr/local/bin/tokenlight-entrypoint"]
CMD ["bash"]
