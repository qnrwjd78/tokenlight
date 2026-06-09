FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG CUDA_ARCH_LIST="8.9"
ARG CONDA_ENV_NAME="tokenlight-wan"
ARG DIFFSYNTH_REF="main"

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST} \
    CONDA_ENV_NAME=${CONDA_ENV_NAME} \
    PATH=/opt/conda/envs/${CONDA_ENV_NAME}/bin:/opt/conda/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/workspace/data/.cache/huggingface \
    MODELSCOPE_CACHE=/workspace/data/.cache/modelscope \
    DIFFSYNTH_MODEL_BASE_PATH=/workspace/models \
    TORCH_HOME=/workspace/data/.cache/torch \
    PIP_CACHE_DIR=/workspace/data/.cache/pip \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

SHELL ["/bin/bash", "-lc"]
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bzip2 \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    libgl1 \
    libglvnd0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    pkg-config \
    wget \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_24.11.1-0-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

RUN conda create -n "${CONDA_ENV_NAME}" -y -c conda-forge python=3.10 pip=25.0 cmake ninja \
    && conda clean -afy

COPY docker/requirements-runtime.txt docker/constraints-runtime.txt /tmp/
RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate "${CONDA_ENV_NAME}" \
    && python -m pip install --upgrade pip==25.0 setuptools==76.0.0 wheel \
    && python -m pip install --no-cache-dir \
        torch==2.6.0 \
        torchvision==0.21.0 \
        torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && python -m pip install --no-cache-dir -c /tmp/constraints-runtime.txt -r /tmp/requirements-runtime.txt \
    && python -m pip install --no-cache-dir "git+https://github.com/modelscope/DiffSynth-Studio.git@${DIFFSYNTH_REF}" \
    && python -c "import importlib.util, torch; assert importlib.util.find_spec('accelerate'); assert importlib.util.find_spec('deepspeed'); assert importlib.util.find_spec('diffsynth'); print('torch', torch.__version__)"

RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc \
    && echo "conda activate ${CONDA_ENV_NAME}" >> /root/.bashrc \
    && echo "cd /workspace" >> /root/.bashrc

COPY docker/entrypoint.sh /usr/local/bin/tokenlight-entrypoint
RUN chmod +x /usr/local/bin/tokenlight-entrypoint

ENTRYPOINT ["/usr/local/bin/tokenlight-entrypoint"]
CMD ["bash"]
