# Fraude - Package d'installation

Package autonome pour deployer la stack Fraude sur une machine Linux deja installee (Ubuntu Server 24.04 LTS recommande, 22.04 fonctionne aussi).

---

## Pre-requis

- **Linux Ubuntu 24.04 (ou 22.04) deja installe** sur la machine cible
- **Acces sudo** sur la machine
- **Internet via ethernet** pendant l'install (pour `apt-get install` de Docker, cockpit, etc.) — ~200 MB
  - WiFi peut etre configure APRES l'install (voir section [Configuration WiFi](#configuration-wifi-apres-install))
- **8 GB RAM minimum**, 50 GB disque libre minimum
- **Cable ethernet** branche au minimum pour l'install (le LAN doit etre le meme que les cameras RTSP du magasin)

---

## Installation (3 commandes)

### 1. Copier le package sur la machine

Depuis ton PC :

```bash
# Option A : via scp (recommande)
scp fraude-package.tar.gz <user>@<ip-laptop>:~/

# Option B : via cle USB
# Copier fraude-package.tar.gz sur la cle, puis sur le laptop :
cp /media/<user>/<usb>/fraude-package.tar.gz ~/
```

### 2. Decompresser et lancer l'installation

```bash
ssh <user>@<ip-laptop>
tar xzf fraude-package.tar.gz
cd fraude-package
sudo ./install.sh
```

### 3. Attendre

Duree totale : ~5-10 min :
- Install des paquets (Docker, cockpit, ufw) : ~2-3 min
- Copie du payload (~3.5 GB) : ~30 sec
- Chargement de l'image Docker (decompression + import) : ~2 min
- Demarrage des containers + verification healthy : ~1-2 min

A la fin, le script affiche les URLs d'acces et la liste des containers.

### 4. (Important) Activer le groupe docker pour ta session SSH

`install.sh` t'ajoute automatiquement au groupe `docker`, mais ta session SSH actuelle n'a pas encore le nouveau groupe. Sinon `docker ps` retourne "permission denied".

```bash
# Soit activer le groupe immediatement :
newgrp docker

# Soit te deconnecter/reconnecter en SSH (groupes recharges au login)
exit
ssh <user>@<ip-laptop>
```

---

## Acces apres installation

| Service | URL | Login |
|---------|-----|-------|
| **Dashboard Streamlit** | http://`<ip>`:8502 | (pas de login) |
| **Cockpit (admin web)** | https://`<ip>`:9090 | `<user>` / mot de passe Linux |
| **MJPEG live camera** | http://`<ip>`:8555/stream | (pas de login) |
| **SSH** | `ssh <user>@<ip>` | mot de passe Linux |

---

## Cameras pre-configurees

La DB SQLite (`/opt/fraude/data/fraude.db`) contient deja **7 cameras du magasin** :

| Camera | Etat | Mode | URL RTSP |
|--------|------|------|----------|
| **Caisse** | ACTIVE | caisse | rtsp://admin:dvr24434@192.168.1.5:554/Streaming/Channels/301 |
| **CAM5** | ACTIVE | vol | rtsp://admin:dvr24434@192.168.1.5:554/Streaming/Channels/501 |
| CAM1 | inactive | vol | rtsp://admin:dvr24434@192.168.1.3:554/Streaming/Channels/101 |
| CAM2 | inactive | vol | rtsp://admin:dvr24434@192.168.1.3:554/Streaming/Channels/201 |
| CAM4 | inactive | vol | rtsp://admin:dvr24434@192.168.1.3:554/Streaming/Channels/401 |
| CAM6 | inactive | vol | rtsp://admin:dvr24434@192.168.1.3:554/Streaming/Channels/601 |
| CAM8 | inactive | vol | rtsp://admin:dvr24434@192.168.1.3:554/Streaming/Channels/801 |

**Pour modifier les URLs RTSP / activer les autres cameras :**
Dashboard > Sidebar > Administration > Onglet **Cameras**

⚠️ Attention CPU : sur i7-8650U (4 cores), max **2-3 cameras actives** simultanement.

---

## Verification post-install

```bash
# Statut du service
sudo systemctl status fraude.service
# Doit afficher "Active: active (exited)"

# Containers
docker compose -f /opt/fraude/docker-compose.yml ps
# Doit afficher fraud-detector + fraud-dashboard en (healthy)

# Logs en temps reel
docker logs -f fraud-detector
docker logs -f fraud-dashboard
sudo journalctl -u fraude.service -f
```

---

## Tunings recommandes pour laptop magasin

A executer une fois l'install validee, en SSH :

```bash
# 1. Desactiver mise en veille a fermeture du capot (laptop dans placard du magasin)
sudo sed -i 's/^#HandleLidSwitch=.*/HandleLidSwitch=ignore/' /etc/systemd/logind.conf
sudo sed -i 's/^#HandleLidSwitchExternalPower=.*/HandleLidSwitchExternalPower=ignore/' /etc/systemd/logind.conf
sudo sed -i 's/^#HandleLidSwitchDocked=.*/HandleLidSwitchDocked=ignore/' /etc/systemd/logind.conf
sudo systemctl restart systemd-logind
# Verifier : grep "HandleLidSwitch" /etc/systemd/logind.conf  → tous "ignore" sans #

# 2. CPU governor performance (au lieu de powersave par defaut, +30% inference YOLO)
sudo apt install -y linux-tools-generic linux-tools-$(uname -r)
sudo cpupower frequency-set -g performance

# Persistance au boot
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl enable --now cpupower.service 2>/dev/null || true

# Verification
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# Doit afficher : performance

# 3. Retention des alertes a 7 jours (au lieu de 30)
# Dashboard > Administration > Onglet Metier > "Retention (jours)" = 7
```

---

## Configuration WiFi (apres install)

L'install se fait obligatoirement en **ethernet** (apt download des paquets). Apres install OK, on peut activer le WiFi pour deplacer le laptop sans cable.

**Identifier l'interface WiFi** :

```bash
ip link show | grep -E "wlp|wlan"
# Sur ThinkPad T480s : typiquement wlp61s0 (Intel AC 8265)
```

**Methode recommandee : nmcli** :

```bash
# Verifier si NetworkManager est installe
nmcli --version 2>/dev/null && echo "OK" || echo "A INSTALLER"

# Si pas installe :
sudo apt update && sudo apt install -y network-manager
sudo systemctl enable --now NetworkManager
echo "network: {config: disabled}" | sudo tee /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg

# Scanner les WiFi disponibles
sudo nmcli device wifi list

# Se connecter (remplace SSID et MOTDEPASSE)
sudo nmcli device wifi connect "TON_SSID" password "TON_MOT_DE_PASSE" ifname wlp61s0

# Verifier
nmcli connection show --active
ip a show wlp61s0
ping -c 3 -I wlp61s0 8.8.8.8
```

⚠️ **L'IP du laptop change** quand tu passes d'ethernet a WiFi (DHCP de la box WiFi != ethernet). Si tu accedes au laptop par IP locale, il faudra te re-decouvrir la nouvelle IP. **Mieux : installer Tailscale** (voir section suivante) — IP fixe dans le tailnet, peu importe le reseau.

---

## Acces remote via Tailscale (recommande)

Pour acceder au dashboard depuis l'exterieur du magasin (bureau, telephone, deplacement), on installe **Tailscale** : VPN mesh peer-to-peer, gratuit (jusqu'a 100 devices), pas de port forwarding necessaire sur la box du magasin. **Bonus** : l'IP Tailscale (100.x.x.x) est fixe — pas besoin de reservation DHCP sur la box.

**Sur le laptop magasin :**

```bash
# Install
curl -fsSL https://tailscale.com/install.sh | sh

# Auth (genere une URL a ouvrir dans navigateur pour login Google/MS/etc)
sudo tailscale up

# Verifier
tailscale ip -4              # IP Tailscale (ex: 100.123.127.5)
tailscale status             # liste des devices du tailnet
```

**Sur les machines admin (PC, telephone) :**
- Telecharger : https://tailscale.com/download
- Installer + login avec **le meme compte** que le laptop magasin
- Tu pourras maintenant ping/SSH/HTTP le laptop magasin

**Acces aux services depuis n'importe quel device du tailnet :**
- Dashboard : http://`<tailscale-ip>`:8502
- Cockpit : https://`<tailscale-ip>`:9090
- MJPEG : http://`<tailscale-ip>`:8555/stream
- SSH : `ssh <user>@<tailscale-ip>`

**MagicDNS (optionnel)** : active sur https://login.tailscale.com/admin/dns pour utiliser `http://fraude:8502` au lieu de l'IP.

**Pas besoin** : IP publique, port forwarding, DDNS, OpenVPN, certificats SSL.

---

## Maintenance

### Redemarrer la stack

```bash
sudo systemctl restart fraude.service
```

### Voir les ressources utilisees

```bash
docker stats fraud-detector fraud-dashboard
```

### Stopper temporairement

```bash
sudo systemctl stop fraude.service
```

### Mise a jour de l'app

Quand on a une nouvelle version :

```bash
# 1. Sauvegarder la DB AVANT
sudo cp /opt/fraude/data/fraude.db ~/fraude.db.backup-$(date +%F)

# 2. Stopper la stack
sudo systemctl stop fraude.service

# 3. Effacer l'ancienne version + le marker du load-image (force re-load nouvelle image)
sudo rm -rf /opt/fraude
sudo rm -f /var/lib/fraude/images-loaded

# 4. Installer la nouvelle version
tar xzf fraude-package-vN.tar.gz
cd fraude-package
sudo ./install.sh

# 5. Restaurer la DB pour conserver les alertes / reglages
sudo systemctl stop fraude.service
sudo cp ~/fraude.db.backup-$(date +%F) /opt/fraude/data/fraude.db
sudo chmod a+rw /opt/fraude/data/fraude.db
sudo systemctl start fraude.service
```

---

## Depannage

### "fraude.service en failed"

```bash
sudo journalctl -u fraude.service -n 100
docker compose -f /opt/fraude/docker-compose.yml logs --tail 50
```

Causes typiques :
- Docker pas encore demarre → `sudo systemctl start docker`
- Conflit de port 8502 ou 8555 → autre service tourne dessus

### "Containers pas healthy" / "fraud-detector restart en boucle"

```bash
docker logs fraud-detector --tail 50
docker logs fraud-dashboard --tail 50
```

Causes typiques :
- **`sqlite3.OperationalError: attempt to write a readonly database`** : permissions Docker volumes (le user `fraude` dans le container a un UID different du user host). Fix :
  ```bash
  sudo chmod -R a+rwX /opt/fraude/data /opt/fraude/recordings /opt/fraude/snapshots /opt/fraude/sounds
  sudo systemctl restart fraude.service
  ```
  *(Note : `install.sh` v2+ fait deja ce chmod automatiquement. Si tu vois ce bug, c'est que tu as un vieux package.)*
- Erreur dans le pipeline (RTSP injoignable, modele ONNX corrompu)
- Pas assez de RAM (verif : `free -h`)

### "permission denied while trying to connect to the Docker daemon socket"

Tu as ete ajoute au groupe docker mais ta session SSH actuelle n'a pas le nouveau groupe. Fix :

```bash
newgrp docker
# OU
exit && ssh <user>@<ip>
```

### "Pas d'acces aux cameras RTSP"

```bash
# Tester ping vers le NVR
ping 192.168.1.5

# Tester le flux RTSP avec ffmpeg dans un container
docker run --rm --network host jrottenberg/ffmpeg \
    -i "rtsp://admin:dvr24434@192.168.1.5:554/Streaming/Channels/301" \
    -t 2 -f null -
```

Si le NVR a une IP differente : modifier via Dashboard > Administration > Cameras.

### "Image Docker corrompue / re-charger l'image"

```bash
sudo systemctl stop fraude.service
sudo rm /var/lib/fraude/images-loaded
sudo systemctl start fraude.service
# load-images.sh va re-decompresser et re-charger l'image
```

### "Bug ThinkPad T480s : laptop boucle sur logo Lenovo apres install Ubuntu"

Bug **Phoenix BIOS T480s** : le BIOS n'utilise pas les entrees NVRAM creees par Linux pour le boot. Solution : **Microsoft Hack** — remplacer `bootmgfw.efi` par notre shim Ubuntu. Procedure complete dans `Fraude/CLAUDE.md` section "Bug critique : Phoenix BIOS T480s".

Resume :
1. Boot sur Ubuntu Desktop Live USB → "Try Ubuntu"
2. `sudo -i` puis :
   ```bash
   mount /dev/nvme0n1p1 /mnt/efi
   mkdir -p /mnt/efi/EFI/Microsoft/Boot
   cp /mnt/efi/EFI/ubuntu/shimx64.efi /mnt/efi/EFI/Microsoft/Boot/bootmgfw.efi
   cp /mnt/efi/EFI/ubuntu/grubx64.efi /mnt/efi/EFI/Microsoft/Boot/grubx64.efi
   cp /mnt/efi/EFI/ubuntu/grub.cfg    /mnt/efi/EFI/Microsoft/Boot/grub.cfg
   cp /mnt/efi/EFI/ubuntu/mmx64.efi   /mnt/efi/EFI/Microsoft/Boot/mmx64.efi
   efibootmgr --create --disk /dev/nvme0n1 --part 1 --label "Windows Boot Manager" --loader "\EFI\Microsoft\Boot\bootmgfw.efi"
   sync && systemctl reboot -i
   ```

---

## Structure du package

```
fraude-package/
├── install.sh                 # Script d'install (a executer en sudo)
├── README.md                  # Ce fichier
├── docker-compose.yml         # Compose avec image: (pas build:)
├── .env                       # Config app (cameras vides → DB)
├── Dockerfile                 # Pour rebuild d'urgence (optionnel)
├── requirements.txt
├── images/
│   └── fraude-images.tar.gz   # Image Docker pre-construite (~3.3 GB)
├── app/                       # Code Python (pipeline)
├── dashboard/                 # Code Streamlit
├── scripts-app/               # Scripts utilitaires de l'app
├── scripts/
│   └── load-images.sh         # Charge l'image au boot (idempotent)
├── sounds/alerte.wav
├── models/                    # ONNX (montes en volume)
│   ├── yolov8n.onnx
│   ├── yolov8n-pose.onnx
│   └── yolov8n-oiv7.onnx
├── data/
│   └── fraude.db              # DB seedee (7 cameras)
└── systemd/
    └── fraude.service         # Unit systemd
```

---

## Securite

⚠️ **Mots de passe par defaut** :
- User Linux : celui que tu as defini pendant l'install Ubuntu
- Telegram bot token : a renseigner via Dashboard > Administration > Alertes

⚠️ **Firewall (ufw)** : seuls les ports 22, 8502, 8555, 9090 sont ouverts. Tout le reste est ferme.

⚠️ **Cockpit** : utilise les comptes Linux du systeme. Verifie que ton mot de passe est fort.

⚠️ **Tailscale** : si installe, l'acces au tailnet protege deja la machine. Pas besoin d'exposer les ports sur internet public.
