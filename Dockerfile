FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
LABEL maintainer="Yibo Lin <yibolin@pku.edu.cn>"
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

# Use bash consistently for RUN steps.
SHELL ["/bin/bash", "-lc"]

ARG CLANGD_MAJOR=18

# Installs system dependencies.
RUN apt-get update \
        && apt-get install -y --no-install-recommends \
        flex \
        cmake \
        curl \
        gnupg \
        ca-certificates \
        libcairo2-dev \
        libboost-all-dev \
        lsb-release \
        software-properties-common \
        && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
                | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
        && UBUNTU_CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME}")" \
        && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/${UBUNTU_CODENAME}/ llvm-toolchain-${UBUNTU_CODENAME}-${CLANGD_MAJOR} main" \
                > /etc/apt/sources.list.d/llvm.list \
        && apt-get update \
        && apt-get install -y --no-install-recommends "clangd-${CLANGD_MAJOR}" \
        && ln -sf "/usr/bin/clangd-${CLANGD_MAJOR}" /usr/local/bin/clangd \
        && rm -rf /var/lib/apt/lists/*

# Installs system dependencies from conda.
RUN conda install -y -c conda-forge bison \
        && conda clean -afy

# Installs python dependencies.
RUN pip install --no-cache-dir \
        pyunpack>=0.1.2 \
        patool>=1.12 \
        matplotlib>=2.2.2 \
        cairocffi>=0.9.0 \
        pkgconfig>=1.4.0 \
        setuptools>=39.1.0 \
        scipy>=1.10.0 \
        numpy>=1.24.0 \
        shapely>=1.8.5

# Default interactive shell for container startup.
CMD ["/bin/bash"]
