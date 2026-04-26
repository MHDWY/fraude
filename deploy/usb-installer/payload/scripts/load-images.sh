#!/bin/bash
# ============================================
# load-images.sh - Charge les images Docker pre-construites
# ============================================
# Idempotent : utilise un marker /var/lib/fraude/images-loaded pour
# ne pas re-importer l'image a chaque demarrage du service.
#
# Au 1er boot apres install : docker load decompresse + importe (~2 min)
# Au 2eme boot et suivants : skip immediat (marker present)
#
# Si l'archive change (nouveau ISO flashe sur la meme machine) :
#   sudo rm /var/lib/fraude/images-loaded && sudo systemctl restart fraude
# ============================================

set -euo pipefail

ARCHIVE="/opt/fraude/images/fraude-images.tar.gz"
MARKER_DIR="/var/lib/fraude"
MARKER="${MARKER_DIR}/images-loaded"

# Verifier que l'archive existe
if [[ ! -f "$ARCHIVE" ]]; then
    echo "[load-images] ERREUR : archive introuvable : $ARCHIVE"
    echo "[load-images] L'image doit etre embarquee dans le payload."
    exit 1
fi

# Skip si deja charge (marker presente la signature de l'archive)
ARCHIVE_SHA=$(sha256sum "$ARCHIVE" | awk '{print $1}')
if [[ -f "$MARKER" ]] && [[ "$(cat "$MARKER")" == "$ARCHIVE_SHA" ]]; then
    echo "[load-images] OK : images deja chargees (marker $MARKER)"
    exit 0
fi

# Verifier que docker tourne
if ! docker info >/dev/null 2>&1; then
    echo "[load-images] ERREUR : docker daemon non accessible"
    exit 1
fi

# Charger l'image
echo "[load-images] Chargement de $ARCHIVE (taille : $(du -h "$ARCHIVE" | awk '{print $1}'))..."
START=$(date +%s)
gunzip -c "$ARCHIVE" | docker load
END=$(date +%s)
echo "[load-images] OK : images chargees en $((END - START))s"

# Lister les images presentes (verification)
docker images --filter "reference=fraude-*" --format "  - {{.Repository}}:{{.Tag}} ({{.Size}})"

# Marquer comme charge (sha de l'archive pour invalidation auto si re-flash)
mkdir -p "$MARKER_DIR"
echo "$ARCHIVE_SHA" > "$MARKER"
echo "[load-images] Marker ecrit : $MARKER"
