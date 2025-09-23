#Dockerfile Update 092325-1
FROM python:3.10-slim

# System deps for networking and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

ENV APP_DIR=/app/runpod/human \
    PYTHONUNBUFFERED=1 \
    SAM_CHECKPOINT_PATH=/models/sam/sam_vit_h_4b8939.pth \
    GROUNDING_DINO_CONFIG=/models/groundingdino/GroundingDINO_SwinT_OGC.py \
    GROUNDING_DINO_WEIGHTS=/models/groundingdino/groundingdino_swint_ogc.pth

WORKDIR /app

# Install runtime deps (torch CPU wheels first, then the rest)
COPY requirements.txt ${APP_DIR}/requirements.txt
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && python -m pip install --no-cache-dir -r ${APP_DIR}/requirements.txt \
 && python -m pip install --no-cache-dir groundingdino-py>=0.1.2 || true

# Download model checkpoints at build time (bake into image)
RUN mkdir -p /models/sam /models/groundingdino \
 && curl -L -o ${SAM_CHECKPOINT_PATH} \
      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
 && curl -L -o ${GROUNDING_DINO_WEIGHTS} \
      https://github.com/IDEA-Research/GroundingDINO/releases/download/groundingdino_swint_ogc/groundingdino_swint_ogc.pth \
 && curl -L -o ${GROUNDING_DINO_CONFIG} \
      https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py

# App code
COPY . ${APP_DIR}

EXPOSE 3000

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","3000","--app-dir","/app/runpod/human"]
