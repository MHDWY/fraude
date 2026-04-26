#!/bin/bash
# ============================================================
# Fraude - Installation post-OS
# ============================================================
# A executer apres avoir installe Ubuntu Server 24.04 (ou 22.04)
# manuellement sur la machine cible.
#
# Usage :
#   tar xzf fraude-package.tar.gz
#   cd fraude-package
#   sudo ./install.sh
#
# Le script :
#   - Installe Docker + cockpit + ufw + sqlite3 (si manquant)
#   - Copie le payload dans /opt/fraude/
#   - Charge l'image Docker (~2 min, idempotent)
#   - Installe et active fraude.service systemd
#   - Configure le firewall (ports 22, 8502, 8555, 9090)
#   - Demarre la stack
#   - Affiche les URLs d'acces
# ============================================================

set -euo pipefail

# ----- Couleurs -----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_ok()      { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# ----- Verifications prealables -----
if [[ $EUID -ne 0 ]]; then
    log_error "Ce script doit etre execute en root : sudo $0"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INSTALL_DIR="/opt/fraude"

# Determine le user qui a sudo (celui qui aura acces a docker)
TARGET_USER="${SUDO_USER:-}"
if [[ -z "$TARGET_USER" ]] || [[ "$TARGET_USER" == "root" ]]; then
    log_warn "Pas de SUDO_USER detecte. Ajoute manuellement ton user au groupe docker apres :"
    log_warn "    sudo usermod -aG docker <ton_user>"
    TARGET_USER=""
fi

# Verifier qu'on est sur Ubuntu/Debian
if ! command -v apt-get &> /dev/null; then
    log_error "Ce script est concu pour Ubuntu/Debian (apt-get manquant)"
    exit 1
fi

# Verifier la presence des fichiers du package
for f in "images/fraude-images.tar.gz" "docker-compose.yml" "systemd/fraude.service" "scripts/load-images.sh"; do
    if [[ ! -f "$SCRIPT_DIR/$f" ]]; then
        log_error "Fichier manquant dans le package : $f"
        log_error "Le package est corrompu ou incomplet."
        exit 1
    fi
done

echo
echo "==============================================="
echo "Fraude - Installation"
echo "==============================================="
echo "  Source     : $SCRIPT_DIR"
echo "  Destination: $INSTALL_DIR"
echo "  User docker: ${TARGET_USER:-<aucun>}"
echo "==============================================="
echo

# ----- 1. Installation des paquets systeme -----
log_info "[1/8] Mise a jour des paquets et installation de Docker..."

apt-get update -qq

NEEDED_PACKAGES=()
command -v docker         &>/dev/null || NEEDED_PACKAGES+=("docker.io")
dpkg -s docker-compose-v2 &>/dev/null || NEEDED_PACKAGES+=("docker-compose-v2")
command -v ufw            &>/dev/null || NEEDED_PACKAGES+=("ufw")
command -v sqlite3        &>/dev/null || NEEDED_PACKAGES+=("sqlite3")
dpkg -s cockpit           &>/dev/null || NEEDED_PACKAGES+=("cockpit")
command -v curl           &>/dev/null || NEEDED_PACKAGES+=("curl")

if [[ ${#NEEDED_PACKAGES[@]} -gt 0 ]]; then
    log_info "  Paquets a installer : ${NEEDED_PACKAGES[*]}"
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${NEEDED_PACKAGES[@]}"
    log_ok "Paquets installes"
else
    log_ok "Tous les paquets sont deja installes"
fi

# ----- 2. Activation des services systeme -----
log_info "[2/8] Activation des services systeme..."
systemctl enable --now docker.service
systemctl enable --now cockpit.socket
log_ok "docker.service et cockpit.socket actives"

# ----- 3. Ajouter l'utilisateur au groupe docker -----
if [[ -n "$TARGET_USER" ]]; then
    log_info "[3/8] Ajout de '$TARGET_USER' au groupe docker..."
    usermod -aG docker "$TARGET_USER"
    log_ok "$TARGET_USER ajoute au groupe docker (deconnecte/reconnecte pour prise en compte)"
else
    log_warn "[3/8] Skip (pas de SUDO_USER)"
fi

# ----- 4. Copie du payload vers /opt/fraude -----
log_info "[4/8] Copie du payload vers $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"

# Copie tout sauf install.sh et README.md
rsync -a --exclude='install.sh' --exclude='README.md' "$SCRIPT_DIR/" "$INSTALL_DIR/"

# Permissions de base
chmod +x "$INSTALL_DIR/scripts/load-images.sh"
if [[ -n "$TARGET_USER" ]]; then
    chown -R "$TARGET_USER:$TARGET_USER" "$INSTALL_DIR"
fi

# IMPORTANT : permissions ecriture sur les volumes monteables.
# Le user 'fraude' DANS le container a un UID different (UID 1001 typiquement)
# du user 'fraude' SUR le host (UID 1000). Sans chmod a+rwX sur ces dossiers,
# le container ne peut pas ecrire la DB SQLite, le heartbeat, les recordings,
# etc. → fraud-detector restart en boucle, fraud-dashboard "readonly database".
log_info "      Permissions ecriture sur volumes (DB, recordings, snapshots, sounds)..."
chmod -R a+rwX "$INSTALL_DIR/data" "$INSTALL_DIR/recordings" "$INSTALL_DIR/snapshots" "$INSTALL_DIR/sounds"

log_ok "Payload copie ($(du -sh "$INSTALL_DIR" | awk '{print $1}'))"

# ----- 5. Installation du service systemd -----
log_info "[5/8] Installation du service systemd fraude.service..."
cp "$INSTALL_DIR/systemd/fraude.service" /etc/systemd/system/fraude.service
systemctl daemon-reload
systemctl enable fraude.service
log_ok "fraude.service installe et active au boot"

# ----- 6. Configuration du firewall -----
log_info "[6/8] Configuration du firewall (ufw)..."
ufw --force reset > /dev/null
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    comment 'SSH'
ufw allow 8502/tcp  comment 'Streamlit dashboard'
ufw allow 8555/tcp  comment 'MJPEG live camera'
ufw allow 9090/tcp  comment 'Cockpit web admin'
ufw --force enable
log_ok "Firewall configure (22, 8502, 8555, 9090 ouverts)"

# ----- 7. Demarrage de fraude.service (load image + compose up) -----
log_info "[7/8] Demarrage de fraude.service..."
log_info "      Etape 1 : load-images.sh decompresse + charge l'image Docker (~2 min)"
log_info "      Etape 2 : docker compose up -d --remove-orphans (~10 sec)"
echo

systemctl start fraude.service &
SERVICE_PID=$!

# Affiche les logs en temps reel pendant le demarrage
journalctl -u fraude.service -f --no-pager &
JOURNAL_PID=$!

# Attend la fin du systemctl start (max 10 min)
wait $SERVICE_PID || true
sleep 5
kill $JOURNAL_PID 2>/dev/null || true

if ! systemctl is-active --quiet fraude.service; then
    log_error "fraude.service n'a pas demarre correctement"
    log_error "Diagnostic : sudo journalctl -u fraude.service -n 50"
    exit 1
fi

log_ok "fraude.service actif"

# ----- 8. Verification de la sante des containers -----
log_info "[8/8] Attente que les containers soient healthy..."

for i in {1..36}; do
    DETECTOR_OK=$(docker inspect --format='{{.State.Health.Status}}' fraud-detector 2>/dev/null || echo "missing")
    DASHBOARD_OK=$(docker inspect --format='{{.State.Health.Status}}' fraud-dashboard 2>/dev/null || echo "missing")

    if [[ "$DETECTOR_OK" == "healthy" ]] && [[ "$DASHBOARD_OK" == "healthy" ]]; then
        log_ok "Les 2 containers sont healthy !"
        break
    fi

    if [[ "$i" == "36" ]]; then
        log_warn "Containers pas encore healthy apres 3 min :"
        log_warn "  fraud-detector  : $DETECTOR_OK"
        log_warn "  fraud-dashboard : $DASHBOARD_OK"
        log_warn "Verifier : docker logs fraud-detector --tail 30"
        break
    fi

    sleep 5
done

# ----- Affichage des URLs -----
IP=$(hostname -I | awk '{print $1}')

echo
echo "==============================================="
echo -e "${GREEN}[OK] Installation Fraude terminee !${NC}"
echo "==============================================="
echo
echo "  Acces depuis n'importe quel PC du LAN :"
echo
echo "  Dashboard Streamlit  : http://$IP:8502"
echo "  MJPEG live camera    : http://$IP:8555/stream"
echo "  Cockpit (admin web)  : https://$IP:9090"
echo "  SSH                  : ssh ${TARGET_USER:-<user>}@$IP"
echo
echo "  Containers :"
docker compose -f "$INSTALL_DIR/docker-compose.yml" ps --format "    {{.Name}}\t{{.Status}}"
echo
echo "  Configuration camera (URLs RTSP) :"
echo "    Dashboard > Sidebar > Administration > Onglet Cameras"
echo
echo "  Logs en temps reel :"
echo "    docker logs -f fraud-detector"
echo "    docker logs -f fraud-dashboard"
echo "    sudo journalctl -u fraude.service -f"
echo
echo "==============================================="

if [[ -n "$TARGET_USER" ]]; then
    echo
    log_warn "IMPORTANT : tu as ete ajoute au groupe docker mais il faut te"
    log_warn "deconnecter/reconnecter (ou faire 'newgrp docker') pour que ca"
    log_warn "soit pris en compte. Sinon : 'docker ps' donnera 'permission denied'."
    echo
fi
