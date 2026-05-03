# ─────────────────────────────────────────────────────────────────────────────
# Gaussian Splatting Docker Image
# Includes: COLMAP (with GUI + CUDA), 3D Gaussian Splatting, SIBR Viewer
# Base: CUDA 11.8 + Ubuntu 22.04
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# ─── Environment ─────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Bratislava
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
# Path to CUDA toolkit (needed by setup.py of gaussian splatting submodules)
ENV CUDA_HOME=/usr/local/cuda
# Build CUDA extensions for common GPU architectures without a GPU present
# Covers: Pascal(6.0), Volta(7.0), Turing(7.5), Ampere(8.0,8.6), Ada(8.9), Hopper(9.0)
ENV TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6 8.9 9.0"
ENV FORCE_CUDA=1

# ─── Base system tools ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    unzip \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# ─── Python 3.10 ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ─── X11 / OpenGL (COLMAP GUI + SIBR viewer) ─────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    libglu1-mesa-dev \
    libglew-dev \
    libglfw3-dev \
    libxxf86vm-dev \
    x11-apps \
    mesa-utils \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxext-dev \
    libxrender-dev \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# ─── COLMAP dependencies ─────────────────────────────────────────────────────
# Building COLMAP 3.9.1 from source for CUDA-accelerated feature matching.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libfreeimage-dev \
    libflann-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libceres-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/colmap/colmap.git /tmp/colmap \
        --branch 3.9.1 --depth 1 && \
    cd /tmp/colmap && mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_ENABLED=ON \
        -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86;89;90" \
        -G Ninja \
    && ninja -j2 \
    && ninja install \
    && rm -rf /tmp/colmap

# ─── SIBR Viewer dependencies ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libassimp-dev \
    libboost-all-dev \
    libgtk-3-dev \
    libopencv-dev \
    libavdevice-dev \
    libavcodec-dev \
    libavutil-dev \
    libavformat-dev \
    libswscale-dev \
    libglm-dev \
    libembree-dev \
    && rm -rf /var/lib/apt/lists/*

# ─── PyTorch 2.0.1 + CUDA 11.8 ───────────────────────────────────────────────
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir \
        torch==2.0.1+cu118 \
        torchvision==0.15.2+cu118 \
        torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cu118

# ─── Gaussian Splatting ───────────────────────────────────────────────────────
RUN git clone --recursive \
    https://github.com/graphdeco-inria/gaussian-splatting \
    /opt/gaussian-splatting

WORKDIR /opt/gaussian-splatting

# Python dependencies (requirements.txt was removed from the repo)
RUN pip3 install --no-cache-dir plyfile tqdm opencv-python-headless "numpy<2"

# Build CUDA extensions – compiled for all listed architectures without GPU.
# --no-build-isolation is required because setup.py imports torch at parse time.
RUN pip3 install --no-cache-dir --no-build-isolation submodules/diff-gaussian-rasterization
RUN pip3 install --no-cache-dir --no-build-isolation submodules/simple-knn
RUN pip3 install --no-cache-dir --no-build-isolation submodules/fused-ssim

# ─── SIBR Viewers (build from source) ────────────────────────────────────────
# Produces binaries in SIBR_viewers/install/bin/
RUN cd SIBR_viewers && \
    cmake -Bbuild . \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja \
    && cmake --build build -j2 --target install

ENV PATH="/opt/gaussian-splatting/SIBR_viewers/install/bin:${PATH}"
# PyTorch native libs must be on the linker path for simple-knn, diff-gaussian-rasterization, etc.
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}"

# ─── Workspace ────────────────────────────────────────────────────────────────
RUN mkdir -p /workspace/data /workspace/output

# Symlinky: z /opt/gaussian-splatting/data → /workspace/data
# Tak príkaz "python3 train.py -s data/truck/" funguje priamo
RUN ln -s /workspace/data   /opt/gaussian-splatting/data && \
    ln -s /workspace/output /opt/gaussian-splatting/output

WORKDIR /opt/gaussian-splatting

VOLUME ["/workspace"]

CMD ["/bin/bash", "--login"]
