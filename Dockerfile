# Optional: run the whole assignment in a container (works the same on Windows,
# macOS and Linux with Docker Desktop).
#   docker build -t mlops-a2 .
#   docker run --rm -v "$PWD/results:/app/results" mlops-a2            # everything
#   docker run --rm -v "$PWD/results:/app/results" mlops-a2 ./run_all.sh b   # one part
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive \
    OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential openmpi-bin libopenmpi-dev python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x run_all.sh part_a_openmp/scaling.sh setup/*.sh && make -C part_a_openmp
CMD ["./run_all.sh"]
