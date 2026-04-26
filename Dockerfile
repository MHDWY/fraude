# ============================================
# Dockerfile - Détection de Fraude Magasin
# Image optimisée pour CPU (pas de GPU requis)
# ============================================

FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Equipe Sécurité Magasin"
LABEL description="Système de détection de fraude par vision par ordinateur"
LABEL version="1.0.0"

# Variables d'environnement pour éviter les interactions
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Installation des dépendances système
# libgl1 et libglib2.0 pour OpenCV
# ffmpeg pour l'encodage vidéo
# pulseaudio pour le son des alertes
# sqlite3 pour la base de données
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    pulseaudio \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Création d'un utilisateur non-root pour la sécurité
RUN groupadd -r fraude && useradd -r -g fraude -m -s /bin/bash fraude

# Répertoire de travail
WORKDIR /opt/fraude

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY app/ ./app/
COPY dashboard/ ./dashboard/
COPY scripts/ ./scripts/
COPY sounds/ ./sounds/

# Création des répertoires de données
RUN mkdir -p /opt/fraude/recordings /opt/fraude/models /opt/fraude/data

# Telecharger et convertir les modeles YOLO en ONNX pendant le build
# - yolov8n.onnx       : detection COCO 80 classes (pipeline principal)
# - yolov8n-pose.onnx  : estimation de pose (17 keypoints)
# - yolov8n-oiv7.onnx  : Open Images V7 ~600 classes (apprentissage zones statiques)
WORKDIR /tmp/yolo_export
RUN python -c "\
from ultralytics import YOLO; \
import shutil, glob; \
m = YOLO('yolov8n.pt'); \
m.export(format='onnx', imgsz=640, dynamic=True); \
m2 = YOLO('yolov8n-pose.pt'); \
m2.export(format='onnx', imgsz=640, dynamic=True); \
m3 = YOLO('yolov8n-oiv7.pt'); \
m3.export(format='onnx', imgsz=640, dynamic=True); \
" && cp /tmp/yolo_export/*.onnx /opt/fraude/models/ && \
    ls -lh /opt/fraude/models/ && \
    rm -rf /tmp/yolo_export
WORKDIR /opt/fraude

# Modele fashion optionnel : placer yolov8n-fashion.onnx dans models/ pour activer
# la detection de vetements (13 classes DeepFashion2). Sans ce fichier, le systeme
# fonctionne normalement sans la detection vetements.
# Pour l'installer: copier le fichier ONNX dans Fraude/models/ avant le build,
# ou monter un volume: -v ./models:/opt/fraude/models

RUN chown -R fraude:fraude /opt/fraude

# Passage à l'utilisateur non-root
USER fraude

# Verification de sante : verifie que le pipeline traite des frames
# Le pipeline ecrit un heartbeat avec timestamp toutes les 30 frames
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import time; from pathlib import Path; \
        hb = Path('/opt/fraude/data/heartbeat'); \
        assert hb.exists(), 'No heartbeat'; \
        ts = float(hb.read_text().split()[0]); \
        age = time.time() - ts; \
        assert age < 120, f'Heartbeat stale: {age:.0f}s'" || exit 1

# Port du dashboard
EXPOSE 8502

# Commande par défaut : lancer le détecteur
CMD ["python", "-m", "app.main"]
