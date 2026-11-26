# Candidate image verified in Docker Hub on 2026-10-24; native build unrun here.
FROM rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0@sha256:cc9b00f90b85c97b015b040fa55c8d1b404b7cacc6ad57d74ee3451c97508da1
ARG INSTALL_TE=1
ENV CI1_ROCM_IMAGE=rocm/pytorch@sha256:cc9b00f90b85c97b015b040fa55c8d1b404b7cacc6ad57d74ee3451c97508da1
ENV GPU_ARCHS=gfx942 PYTORCH_ROCM_ARCH=gfx942 AITER_USE_SYSTEM_TRITON=1
ENV AITER_MAX_JOBS=8 MAX_JOBS=8 OMP_NUM_THREADS=4
RUN apt-get update && apt-get install -y --no-install-recommends git cmake ninja-build libnuma-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/ci1
COPY pyproject.toml ./
COPY terminus ./terminus
COPY production ./production
RUN RUNTIME_ROOT=/opt/ci1-amd-runtime INSTALL_TE=${INSTALL_TE} bash production/amd/build_runtime.sh
ENV PATH=/opt/ci1-amd-runtime/venv/bin:${PATH}
ENV NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1
ENV AITER_JIT_DIR=/opt/ci1-amd-runtime/cache/aiter TRITON_CACHE_DIR=/opt/ci1-amd-runtime/cache/triton
# Explicit human entry points are in README.md; no training at container start.
CMD ["bash"]
