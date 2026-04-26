# Cle USB autoinstall — Fraude

ISO bootable Ubuntu Server 24.04 LTS qui installe **toute la stack Fraude** sur une machine cible (PC laisse en magasin), en moins de 15 min, sans intervention.

---

## Ce qui se passe quand on boot la cle

1. Ubuntu Server 24.04 s'installe automatiquement (cloud-init / autoinstall)
2. Disque cible **entierement efface**, partitionne en LVM ext4
3. Compte cree : `fraude` / `ASX@admin` (ou autre passe en argument du build)
4. Paquets installes : `docker.io`, `docker-compose-v2`, `cockpit`, `openssh-server`, `ufw`
5. Le payload Fraude est copie dans `/opt/fraude/` :
   - Code (`app/`, `dashboard/`, `Dockerfile`, `docker-compose.yml`)
   - Modeles ONNX pre-construits (`models/yolov8n*.onnx`)
   - DB SQLite seedee avec 2 cameras (Caisse active, CAM5 inactive)
   - Service systemd `fraude.service`
6. La machine reboot
7. Au boot : `fraude.service` lance `docker compose up -d --build`
8. Dans 5-10 min (premier build de l'image) : tout tourne

---

## Materiel necessaire

| Item | Detail |
|------|--------|
| Cle USB | **32 GB minimum**, USB 3.0 recommande |
| ISO Ubuntu Server 24.04 LTS | A telecharger sur https://releases.ubuntu.com/24.04/ |
| Machine de build (Windows) | Avec Docker Desktop demarre + PowerShell |
| Machine cible (magasin) | Boot UEFI ou BIOS, 8 GB RAM mini, 50 GB SSD mini, NIC ethernet |
| Cameras | Hikvision RTSP accessibles depuis le LAN (defaut: 192.168.1.5) |

---

## Etape 1 — Construire l'ISO sur Windows

Depuis PowerShell, dans le repo Fraude :

```powershell
# Pre-requis : Docker Desktop demarre
cd "C:\AI\Claude Projects\Textile et retail\Fraude"

# Mot de passe par defaut : ASX@admin
.\deploy\usb-installer\build-iso.ps1 `
    -UbuntuIso "C:\Downloads\ubuntu-24.04.1-live-server-amd64.iso"

# Ou avec un mot de passe custom
.\deploy\usb-installer\build-iso.ps1 `
    -UbuntuIso "C:\Downloads\ubuntu-24.04.1-live-server-amd64.iso" `
    -OutputIso "D:\fraude-installer.iso" `
    -Password "MonMotDePasse!"
```

Duree : ~3-5 min (depend du download de `ubuntu:24.04` la 1ere fois).

Sortie : `fraude-installer.iso` (~1.5 GB).

---

## Etape 2 — Flasher la cle USB

### Avec **Rufus** (Windows, recommande)

1. Telecharger https://rufus.ie/
2. Brancher la cle USB **32 GB** (toutes les donnees seront effacees !)
3. Lancer Rufus :
   - **Peripherique** : la cle USB
   - **Selection de boot** : `fraude-installer.iso`
   - **Schema de partition** : GPT (UEFI moderne) ou MBR (vieux PC)
   - **Mode d'image** : choisir **"Mode image DD"** quand Rufus le demande
4. Cliquer DEMARRER, attendre 5-10 min

### Avec `dd` (Linux/macOS)

```bash
# Identifier la cle (sda? sdb? mmcblk0?)
lsblk

# Flasher (REMPLACER /dev/sdX par la bonne lettre !)
sudo dd if=fraude-installer.iso of=/dev/sdX bs=4M status=progress conv=fsync
sync
```

---

## Etape 3 — Installer sur la machine du magasin

1. Brancher la cle USB sur la machine cible
2. Demarrer la machine, entrer dans le BIOS (touche F2 / F12 / Suppr selon le PC)
3. **Boot order** : mettre la cle USB en premier
4. Sauvegarder, redemarrer
5. Le menu GRUB s'affiche brievement (1 sec) puis **autoinstall demarre tout seul**
6. **NE PAS toucher au clavier** pendant l'install (~10-15 min)
7. La machine **reboot automatiquement** quand l'install est finie
8. Au reboot : Ubuntu boote sur le SSD, `fraude.service` demarre `docker compose`
9. Premier `docker compose up --build` : ~5-10 min (build de l'image fraude-detector)
10. **Debrancher la cle USB** une fois le 1er reboot fait

---

## Etape 4 — Verification post-install

Trouver l'IP de la machine (ecran ou via la box):

```bash
# Sur la machine du magasin (ecran/clavier branche)
ip addr show
# Ou se connecter via SSH depuis un autre PC :
ssh fraude@<ip>
# Mot de passe : ASX@admin
```

### Acces depuis n'importe quel PC du LAN

| Service | URL | Login |
|---------|-----|-------|
| **Dashboard Streamlit** | http://`<ip>`:8502 | (pas de login) |
| **Cockpit (admin web)** | https://`<ip>`:9090 | fraude / ASX@admin |
| **MJPEG live camera** | http://`<ip>`:8555/stream | (pas de login) |
| **SSH** | `ssh fraude@<ip>` | fraude / ASX@admin |

### Verifier que tout tourne

```bash
ssh fraude@<ip>
sudo systemctl status fraude.service       # doit etre "active (exited)"
docker compose -f /opt/fraude/docker-compose.yml ps    # 2 containers UP healthy
docker logs fraud-detector --tail 50       # supervisor + workers
docker logs fraud-dashboard --tail 20      # streamlit ready
```

---

## Etape 5 — Configuration cameras (apres install)

Les 2 cameras sont pre-configurees dans la DB :
- **Caisse** (ACTIVE) — `rtsp://admin:dvr24434@192.168.1.5:554/Streaming/Channels/102`
- **CAM5** (inactive) — `rtsp://admin:dvr24434@192.168.1.5:554/Streaming/Channels/502`

Si l'IP du NVR ou les credentials sont differents au magasin :

1. Aller sur http://`<ip>`:8502
2. Sidebar > **Administration** > Onglet **Cameras**
3. Editer chaque camera (URL, zone, niveau, mode)

Pour activer la 2eme camera : cocher "active" sur CAM5 (attention performances CPU).

---

## Maintenance distante

### Via SSH

```bash
ssh fraude@<ip>
# Voir les logs en temps reel
docker logs -f fraud-detector
# Redemarrer la stack
sudo systemctl restart fraude.service
# Mettre a jour le code (git pull si repo present, ou re-flash la cle)
```

### Via Cockpit (interface web)

`https://<ip>:9090` — donne acces a :
- Etat systeme (CPU, RAM, disque, reseau)
- Journaux (systemd, docker)
- Terminal web
- Gestion services (start/stop/restart fraude.service)
- Mises a jour

---

## Depannage

### "L'install reboote en boucle sur la cle USB"

- BIOS : verifier que le SSD est bien le 1er boot APRES install
- Debrancher la cle apres le 1er reboot

### "fraude.service est en failed"

```bash
sudo journalctl -u fraude.service -n 100
docker compose -f /opt/fraude/docker-compose.yml logs
```

Cause typique : conflit de port ou Docker pas encore pret. Re-essayer :
```bash
sudo systemctl restart fraude.service
```

### "Pas d'acces aux cameras RTSP"

```bash
# Tester la connectivite reseau depuis la machine
ping 192.168.1.5
# Tester le flux RTSP avec ffmpeg
docker run --rm jrottenberg/ffmpeg -i "rtsp://admin:dvr24434@192.168.1.5:554/Streaming/Channels/102" -t 2 -f null -
```

Si le NVR a une IP differente : modifier via Dashboard > Administration > Cameras.

### "Je veux re-installer from scratch"

Re-booter sur la cle USB, l'autoinstall recommence et efface tout.

---

## Securite

⚠️ **Mot de passe par defaut** : `ASX@admin` — **a changer en production** :

```bash
ssh fraude@<ip>
passwd        # changer son propre mot de passe
```

⚠️ **Telegram bot token** : laisse vide dans le `.env`. A renseigner via Dashboard > Administration > Alertes une fois le bot cree.

⚠️ **Firewall (ufw)** : ports ouverts apres install : 22 (SSH), 8502 (Streamlit), 8555 (MJPEG), 9090 (Cockpit). Tout le reste est ferme.

---

## Structure du payload

```
/opt/fraude/                        (sur la machine installee)
├── app/                            # Code pipeline (PipelineFraude, workers)
├── dashboard/                      # Code Streamlit
├── scripts-app/                    # Scripts utilitaires (download_models.py)
├── scripts/                        # Scripts d'install (seed-fraude-db.py)
├── sounds/alerte.wav
├── models/
│   ├── yolov8n.onnx               # Detection COCO
│   ├── yolov8n-pose.onnx          # Pose 17 keypoints
│   └── yolov8n-oiv7.onnx          # Open Images V7
├── data/fraude.db                  # SQLite seedee (2 cameras)
├── recordings/                     # Vide au depart
├── snapshots/                      # Vide au depart
├── systemd/fraude.service          # (copie aussi a /etc/systemd/system/)
├── .env
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
