# Stage 1: Builder with full CUDA toolkit
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY . .

# Stage 2: Runtime image
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# NVIDIA specific environment variables
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.8"

# Copy installed packages and app code
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app

# Set environment paths
ENV PATH=/root/.local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

WORKDIR /app

# Проверка CUDA (опционально, можно оставить в CMD)
RUN python -c "import torch; \
    print(f'PyTorch version: {torch.__version__}'); \
    print(f'CUDA available: {torch.cuda.is_available()}'); \
    print(f'CUDA device: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('⚠️ No GPU detected')"

CMD ["python", "server.py"]
