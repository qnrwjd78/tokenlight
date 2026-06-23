FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    DIFFSYNTH_MODEL_BASE_PATH=/workspace/model

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ffmpeg \
    git \
    libasound2 \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libsndfile1 \
    ninja-build \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python-is-python3 \
    tar \
    tmux \
    unzip \
    xz-utils \
    zip \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r /tmp/requirements.txt \
    && python3 -c "import tensorboard; from torch.utils.tensorboard import SummaryWriter; print('tensorboard ok')"

CMD ["bash"]
