# Reproducible environment for the whole project: Python + pinned packages,
# Quarto, and the project package. The analysis code and data are NOT baked
# into the image: the repository is mounted at /project, so the container
# provides the environment and the repository provides the analysis.
#
#   docker build -t jse-microstructure .
#   docker run --rm -v "$PWD":/project jse-microstructure make synthetic   # no licensed data needed
#   docker run --rm -v "$PWD":/project jse-microstructure make real        # needs data/real/raw/*.gz
#
# For bit-for-bit rebuilds pin the base image by digest:
#   docker buildx imagetools inspect python:3.13.18-slim-bookworm

FROM python:3.13.18-slim-bookworm

ARG QUARTO_VERSION=1.8.25

ENV LANG=C.UTF-8 \
    TZ=UTC \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    NUMBA_CACHE_DIR=/tmp/numba \
    PIP_NO_CACHE_DIR=1

# 1. System tools: make, curl (for Quarto), git (Snakemake metadata)
RUN apt-get update && apt-get install -y --no-install-recommends make curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# 2. Quarto, pinned
RUN curl -fsSL -o /tmp/quarto.deb \
      "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb" \
    && apt-get update && apt-get install -y --no-install-recommends /tmp/quarto.deb \
    && rm -rf /tmp/quarto.deb /var/lib/apt/lists/*

# 3. Python packages, exact versions (changes rarely -> cached layer)
COPY requirements-lock.txt /tmp/requirements-lock.txt
RUN pip install -r /tmp/requirements-lock.txt

# 4. The project package (editable, so the mounted source is what runs)
WORKDIR /project
COPY pyproject.toml LICENSE ./
COPY src ./src
RUN pip install --no-deps -e . && python -c "import jsemicro, numba; print('jsemicro', jsemicro.__version__)"

CMD ["make", "help"]
