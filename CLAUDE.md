# FRAUDE - Retail Fraud Detection System

## Quick Start

```bash
docker compose up -d --build
# Dashboard: http://localhost:8502
```

## Deployment Status (Prod Magasin)

> **Note pour les futures sessions Claude** : cette section reflete l'etat de la prod actuelle. A mettre a jour quand quelque chose change (deploy nouveau site, changement d'IP, ajout camera, etc.)

| Element | Valeur |
|---------|--------|
| Hardware | ThinkPad T480s, i7-8650U @ 1.90 GHz, 16 GB RAM, NVMe Intel 256 GB |
| BIOS | Phoenix SecureCore 1.49 (2021) — **bug NVRAM**, contourne via Microsoft Hack |
| OS | Ubuntu Server 24.04 LTS (install manuel via cle USB GPT/UEFI) |
| Hostname | fraude |
| User Linux | fraude |
| Methode deployment | Package `fraude-package.tar.gz` (3.3 GB) + `install.sh` |
| Date deployment initial | 26 avril 2026 |
| WiFi interface | wlp61s0 (Intel AC 8265) |
| Reseau actuel | WiFi (ethernet seulement utilise pour install initial) |
| IP locale (LAN WiFi) | dynamique via DHCP (peut changer apres reboot) |
| **IP Tailscale (fixe)** | **100.123.127.5** |
| Hostname Tailscale | fraude |
| Acces remote | Tailscale (gratuit, peer-to-peer, pas de port forwarding) |
| Cameras configurees | 7 dans `data/fraude.db` |
| Cameras actives | 2 (Caisse en mode caisse + CAM5 en mode vol) |
| Cameras inactives | 5 (CAM1, CAM2, CAM4, CAM6, CAM8) — a activer selon CPU |
| NVR magasin | Hikvision sur 192.168.1.5 (Caisse) et 192.168.1.3 (autres) |
| Microsoft Hack applique | Oui (`/EFI/Microsoft/Boot/bootmgfw.efi` = shimx64.efi Ubuntu) |
| Tunings appliques | Lid switch ignore + CPU governor performance + cpupower.service |
| Telegram bot | **Configure** : @Frdclt001_bot (id 8595839140) → chat `6119440920` (Mh Mj). Token + chat_id stockes dans `parametres` table cat `telegram` (pas dans `.env`). |
| Repo source code | **https://github.com/MHDWY/fraude** (prive) — deploy key SSH `~/.ssh/github_deploy` sur magasin, read-only |
| Workflow correctifs | Volume mount `/home/fraude/fraude-src/{app,dashboard}` → `/opt/fraude/{app,dashboard}` (ro), patch via `git pull` + `docker compose restart` (~30s, no rebuild) |
| Admin dashboard | Onglet protege par mot de passe (param DB `admin_password`, defaut `asx`, modifiable via Systeme tab) |

**Acces post-deployment :**
- Dashboard : http://100.123.127.5:8502 (ou http://fraude:8502 si MagicDNS active)
- Cockpit : https://100.123.127.5:9090 (login Linux : `fraude` / `asx`)
- SSH : `ssh fraude` (alias dans `~/.ssh/config`, **auth par cle ed25519 — passwordless**)
- MJPEG : http://100.123.127.5:8555/stream

**Setup SSH (deja fait sur le poste mhilali Windows, 26 avril 2026) :**
- Cle locale : `~/.ssh/id_ed25519` (sans passphrase)
- Cle publique deployee dans `/home/fraude/.ssh/authorized_keys` sur le magasin
- Alias `~/.ssh/config` :
  ```
  Host fraude
      HostName 100.123.127.5
      User fraude
      IdentityFile ~/.ssh/id_ed25519
      IdentitiesOnly yes
      ServerAliveInterval 30
  ```
- Si nouveau poste : `ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519` puis copier la cle pub dans `authorized_keys` du magasin (premiere fois via password `asx`).

## Workflow correctifs (git-based, sans rebuild d'image)

**Repo source de verite** : https://github.com/MHDWY/fraude (prive, branche `main`)

Le code Python est embarque dans l'image Docker, mais on monte le dossier `app/` et `dashboard/` du repo cloud en volume read-only sur le magasin. Resultat : pousser un correctif = `git push` cote dev + `git pull` cote magasin + `docker compose restart`. Pas de rebuild, pas de transfert d'image, ~30 sec total.

**Layout sur le magasin :**
```
/opt/fraude/                              # install package (gere par install.sh)
├── docker-compose.yml                    # version magasin (image: pre-construite)
├── docker-compose.override.yml           # LOCAL au magasin, NON commite
│                                         # → mount /home/fraude/fraude-src/app et /dashboard
├── .env, data/, recordings/, snapshots/, sounds/, models/

/home/fraude/fraude-src/                  # git clone read-only de github.com/MHDWY/fraude
├── app/, dashboard/, scripts/, deploy/, ...

/etc/systemd/system/fraude.service.d/
└── override.conf                         # drop-in : retire `-f docker-compose.yml` explicite
                                          # pour que docker compose auto-decouvre l'override.yml
```

**Deploy key GitHub** : la cle SSH `~/.ssh/github_deploy` sur le magasin est enregistree comme Deploy Key (read-only) sur le repo. Permet `git pull` sans interaction. Si nouveau magasin : generer une nouvelle paire et l'ajouter dans Settings → Deploy Keys.

**Workflow standard :**
```powershell
# Cote dev Windows
.\deploy\hotfix.ps1                       # push + pull + restart les 2 containers
.\deploy\hotfix.ps1 -Service detector     # restart fraud-detector uniquement
.\deploy\hotfix.ps1 -NoRestart            # juste pousser le code, restart manuel ensuite
```

Le script `deploy/hotfix.ps1` :
1. Refuse si working tree pas commit (propose commit)
2. `git push`
3. `ssh fraude "cd ~/fraude-src && git pull"` (affiche les commits appliques)
4. `ssh fraude "cd /opt/fraude && docker compose restart <service>"`
5. Wait healthcheck (max 60s) + report etat final

**Workflow manuel (si tu veux tout faire toi-meme) :**
```bash
# Local
git push

# Magasin
ssh fraude
cd ~/fraude-src && git pull
cd /opt/fraude && docker compose restart fraud-detector  # ou dashboard, ou rien (les 2)
docker ps   # check healthy
```

**Rollback :**
```bash
# Local : revenir a l'avant-dernier commit
git revert HEAD && git push
ssh fraude "cd ~/fraude-src && git pull && cd /opt/fraude && docker compose restart"
```

**Limites a connaitre :**
- Le `.env`, la DB, les modeles ONNX ne sont **pas** dans git → modifs de config se font via le dashboard Admin ou directement sur le magasin.
- Si tu modifies `Dockerfile`, `requirements.txt` ou les modeles → necessite un rebuild d'image complet (workflow package installer, pas hotfix).
- Si le package installer (`fraude-package.tar.gz`) est re-deploye, il ecrase `/opt/fraude/` → l'override.yml est perdu, il faut le recreer (les patches du code restent dans `/home/fraude/fraude-src/` qui n'est pas touche par le package).
- L'override.yml et le drop-in systemd ne sont pas dans git → si on remonte un nouveau magasin il faudra les recreer (a integrer dans `install.sh` plus tard).

## Architecture

Real-time video pipeline: Frame → YOLO Detection → ByteTrack → Pose Estimation → Behavior Analysis + Caisse Analysis → Alerts

- **Detection**: YOLOv8-nano + YOLOv8-nano-pose + optional Fashion YOLO (ONNX, CPU-only)
- **Tracking**: ByteTrack with Hungarian algorithm (scipy)
- **Caisse Analyzer**: State machine per transaction (scan → payment → ticket → handoff)
- **Multi-Camera**: `OrchestrateurMultiCamera` spawns one `CameraWorker` thread per active camera in DB
- **Database**: SQLite with WAL mode (requires read-write mount, no `:ro`)
- **Dashboard**: Streamlit on port 8502 (3 pages: Dashboard, Historique, Administration — onglet Administration protege par mot de passe `admin_password`, defaut `asx`)
- **Docker**: 2 services (`fraud-detector`, `fraud-dashboard` with host network)

### Multi-Camera Architecture

- Default mode: `python -m app.main` → multi-camera via `OrchestrateurMultiCamera`
- Legacy mode: `python -m app.main --single` or `--source` or `--test-webcam` → mono-source `PipelineFraude`
- YOLO models loaded once and shared across all workers (thread-safe read-only inference)
- `threading.Semaphore(inference_concurrency)` limits concurrent ONNX inference (default: 4)
- Each `CameraWorker` has its own: ByteTracker, AnalyseurComportements, AnalyseurCaisse, EnregistreurVideo
- Each worker passes its own `enregistreur` to alert manager (video clips use the worker's frame buffer, not a global empty one)
- Supervisor in main thread monitors worker health, auto-relaunches dead workers
- Config: `max_cameras` (default 8), `inference_concurrency` (default 4), `supervisor_interval_seconds` (default 10)

### RTSP Streaming (Hikvision HEVC)

- **Dedicated grab thread** per camera: `cap.read()` in loop, stores latest frame in `self._derniere_frame`
- MUST use `cap.read()` (not `cap.grab()/retrieve()`) — HEVC streams produce corrupt frames with grab/retrieve
- FFmpeg options: `rtsp_transport;tcp|probesize;1000000|analyzeduration;1000000|max_delay;0|reorder_queue_size;0|fflags;nobuffer+discardcorrupt|flags;low_delay`
- MJPEG streaming server on port 8555 for dashboard live view (frame skip: YOLO every 3rd frame)

### CPU Performance Constraints

- Each frame = 2 YOLO inferences (detection + pose) at ~200-400ms each
- **7 cameras + semaphore=2 → ~0.5 fps/camera** (too slow for behavior detection)
- **2 cameras + semaphore=4 → ~3 fps/camera** (workable)
- Current config: 2 active cameras (Caisse + CAM5), inference_concurrency=4
- Future: migrate to GPU laptop for full 7-camera support

### Replay Mode (Historique > Alertes production)

- Select an alert with a video clip → frame-by-frame slider navigation
- Quick nav buttons: start, -1s, +1s, end
- Re-analyze any frame with YOLO + Pose at adjustable confidence
- Video path resolution: tries absolute, relative, extension variants, and subdirectory search

## Key Files

- `app/main.py` — `PipelineFraude` (mono-source) + `OrchestrateurMultiCamera` (multi-camera) + CLI args
- `app/camera_worker.py` — `CameraWorker`: per-camera thread with grab thread, mannequin filter, isolated tracker/analyzer/recorder
- `app/config.py` — Pydantic settings (includes `max_cameras`, `inference_concurrency`)
- `app/database.py` — SQLite: alertes, cameras, parametres, utilisateurs_alertes, etc.
- `app/detector.py` — YOLO ONNX detection + pose estimation (hands, elbows, shoulders, hips) + optional fashion detection
- `app/behavior_analyzer.py` — 2 shoplifting behaviors with DB-configurable parameters and time-normalized score accumulation
- `app/caisse_analyzer.py` — 4 cash register fraud behaviors (state machine)
- `app/alert_manager.py` — Alerts: Telegram (photo + video), Sound, Video recording
- `app/video_recorder.py` — Thread-safe recording with circular buffer, auto FPS estimation from real capture rate
- `dashboard/app.py` — Streamlit: Dashboard, Historique (with replay), Administration (7 tabs)
- `dashboard/live_camera.py` — Persistent RTSP camera viewer with Event-based frame capture + MJPEG server

## Key Conventions

- All monetary values in **Dirhams (DH)**
- All config stored in `parametres` table (key/value with category/type)
- Cameras stored in `cameras` table (name, url, zone, **niveau**, active)
- Settings seeded on first run via `initialiser_parametres_defaut()`
- Config hierarchy: .env → Pydantic Settings → DB parametres → defaults
- Dashboard sidebar: 3 pages (Dashboard, Historique, Administration)
- Admin page: **7 tabs** (Cameras, **Entrainement**, Alertes, Utilisateurs, Detection Vol, Metier, Systeme), protege par mot de passe (`admin_password` en DB cat `systeme`, defaut `asx`, deverrouillage par session via `st.session_state["admin_unlocked"]`, bouton verrouiller manuel, formulaire de changement dans Systeme tab, valeur masquee dans la table "Tous les parametres")
- `st.form()` used for all admin inputs to avoid Streamlit reruns

## Database Tables

- `alertes` — production detection events (bbox, confidence, snapshot, video path) — CRUD via Historique
- `stats_journalieres` — daily aggregates (alerts, loss DH, incidents)
- `parametres` — all configurable settings (~50 defaults : vol, exclusion, telegram, systeme dont `admin_password`)
- `cameras` — registered camera sources (name, url, zone, **niveau**, active, mode_detection)
- `zones_exclusion` — per-camera exclusion zones (pct coordinates, manual/auto source, active flag)
- `sessions_apprentissage` — auto-learning sessions (camera_id, duration, status)
- `zones_proposees` — proposed zones from auto-learning (pending user validation)
- `utilisateurs_alertes` — Telegram alert recipients
- `camera_utilisateurs` — many-to-many camera ↔ recipient routing

## 6 Behaviors Detected

**Shoplifting (2) — Score accumulation in `app/behavior_analyzer.py`:**
1. CACHER_ARTICLE (0.9) — hand in dissimulation zone (hips/waist) + inward movement pattern
2. DISSIMULER_SAC (0.95) — hand-bag proximity + alternation pattern

**Cash Register Fraud (4) — State machine in `app/caisse_analyzer.py`:**
3. SCAN_SANS_TICKET (0.8) — item scanned but no ticket printed
4. PAIEMENT_SANS_TICKET (0.85) — payment detected, no ticket within timeout (12s)
5. DEPART_SANS_TICKET (0.9) — customer leaves checkout without receiving ticket
6. TICKET_SANS_CLIENT (0.85) — ticket printed but no client detected at counter (phantom transaction)

### Behavior Detection Parameters (DB category "vol")

All configurable via Admin > Detection Vol tab:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vol_distance_main_corps` | 0.25 | Max hand-body distance (ratio of bbox height) |
| `vol_zone_dissimulation_haut` | 0.3 | Top of dissimulation zone (0=top of bbox) |
| `vol_zone_dissimulation_bas` | 0.85 | Bottom of dissimulation zone (1=bottom of bbox) |
| `vol_increment_main_corps` | 0.12 | Score increment when hand in dissimulation zone |
| `vol_increment_mouvement_rentrant` | 0.15 | Score increment for inward hand movement |
| `vol_increment_objet_proche` | 0.10 | Bonus when YOLO object near hand |
| `vol_ratio_rapprochement` | 0.6 | dist_after/dist_before ratio for inward detection |
| `vol_historique_min_frames` | 5 | Min observations for movement analysis |
| `vol_sac_increment_base` | 0.10 | Score for hand near bag |
| `vol_sac_increment_alternance` | 0.15 | Bonus for hand-shelf-bag alternation |
| `vol_sac_alternances_min` | 2 | Min alternations for bonus |
| `vol_sac_distance_ratio` | 0.8 | Max hand-bag distance (ratio of bag size) |
| `vol_decay_rate` | 0.95 | Score decay per reference frame |
| `vol_fps_ref` | 2.0 | Reference FPS for decay normalization |

### Score Accumulation

- Decay: `score *= decay_rate ^ (dt * fps_ref)` — time-normalized, FPS-independent
- fps_ref=2.0: at 1fps → decay 0.90/frame, at 15fps → decay 0.99/frame
- Alert threshold: `behavior_threshold` (DB, default 0.4)
- Score global: MAX(score × weight), +15% bonus per concurrent behavior above 0.25
- Cooldown: `alert_cooldown_seconds` (DB, default 30s)

**Caisse state machine:** INACTIF → SCAN → PAIEMENT → ATTENTE_TICKET → TICKET_IMPRIME → REMIS → OK

### Mannequin Filtering

1. **Calibration-based**: IoU overlap with reference objects tagged "mannequin" in DB
2. **Position-based immobility**: tracks static positions across track ID changes, filters after 30s (`mannequin_seuil_immobilite_sec`)
3. Immobility uses real time (not frame count) — works correctly at any FPS

### Zones d'Exclusion (Per-Camera Spatial Filtering)

Per-camera rectangular zones where ALL detections (persons + objects) are ignored. Coordinates stored as percentages (0.0-1.0) for resolution independence.

**Pipeline position**: After mannequin filter, BEFORE tracking (step 1c in `_traiter_frame`)
**Filtering method**: Center-point containment (detection center inside zone → filtered)
**Reload**: Every 300 frames from DB (no restart needed after dashboard changes)

**Creation methods (all via Admin > Entrainement tab):**
1. **Manual coordinates**: Section D — Coordonnees (X/Y min/max in %)
2. **From YOLO detection**: Section B — Capture frame → `detecter_tout_coco()` (80 classes, conf 0.20) → dropdown role assignment (zone_exclusion, mannequin, imprimante, scanner, terminal_paiement, autre)
3. **Auto-learning**: Section E — System observes camera for N minutes, identifies static objects, proposes zones

**Auto-learning flow:**
1. Dashboard creates `sessions_apprentissage` row (statut='en_cours')
2. CameraWorker detects session in periodic maintenance (every 300 frames)
3. Worker collects observations: spatial matching (distance < seuil_px), sliding center update (0.9/0.1)
4. After duration expires: filters observations with duration >= `apprentissage_seuil_immobilite_sec` (default 120s)
5. Writes to `zones_proposees` table, marks session 'terminee'
6. Dashboard displays proposals with Accept/Reject buttons
7. Accepted zones are copied to `zones_exclusion` with source='auto'

**DB parameters (category "exclusion"):**
- `apprentissage_duree_minutes`: 5.0 — default observation duration
- `apprentissage_seuil_immobilite_sec`: 120.0 — min immobility to propose zone
- `apprentissage_seuil_deplacement_px`: 30 — max displacement for "immobile"
- `apprentissage_confiance_min`: 0.3 — min YOLO confidence for observation

### ByteTrack Tuning

- `seuil_score_haut`: 0.3 (lowered from 0.5 for CCTV with partial occlusion)
- Track activation: immediate (1 update, not 3)
- "perdu" tracks kept in analysis for up to 10 frames (intermittent detection tolerance)
- YOLO `CLASSES_PERTINENTES`: personne(0), sac_a_dos(24), sac_a_main(26), valise(28), telephone(67)

## Alert Channels

- **Telegram**: Photo snapshot (immediate) + Video clip (~30s later, via sendDocument)
- **Sound**: 880Hz sine wave (non-blocking)
- **Video**: 30s clip with 5s pre-event buffer, **auto FPS estimation** from real capture rate (not hardcoded 15fps)

## Store Layout

### Floor Levels (Niveaux)
Niveau -1, Niveau 0, Niveau 1, Niveau 2 — assigned per camera in DB

### 9-Zone Grid
```
Rayon_A  Rayon_B  Rayon_C   (top 30%)
Allee_G  Allee_C  Allee_D   (middle 40%)
Entree   Caisse   Sortie    (bottom 30%)
```

## Important Notes

- SQLite volume must NOT be mounted read-only (WAL needs write access for -wal/-shm files)
- YOLO models exported to ONNX during Docker build
- Non-root user `fraude` inside container
- Max 5 persons for pose estimation per frame (CPU constraint)
- RTSP capture: persistent connections, `threading.Event`-based frame signaling, TCP transport
- DEBUG logging active in `camera_worker.py` every 30 frames (to be removed after validation)
- **Hot-reload cameras NON supporte** : modifier `cameras` table (URL, mode_detection, active toggle) via Admin > Cameras ne se propage PAS aux workers en cours. Chaque worker est cree au demarrage de `OrchestrateurMultiCamera` avec sa config DB et la garde jusqu'au crash/restart. Apres modif → restart `fraud-detector` (`docker compose restart fraud-detector` ou via le script hotfix). TODO : ajouter un bouton "Recharger les cameras" dans le dashboard qui envoie un signal au superviseur.
- **Token DVR visible dans les logs** : quand un worker echoue a ouvrir un RTSP, l'URL complete (avec `user:pass@`) est loggee en clair via `logger.error`. A masquer dans `camera_worker.py` (sed-like sur `:[^@]*@` avant log).

### Clothing Detection (Optional)

- `DetecteurVetements` in `app/detector.py` — secondary YOLO model for 13 clothing classes
- Model file: `models/yolov8n-fashion.onnx` (trained on DeepFashion2, not included — must be provided)
- 13 classes: t-shirt, pull, veste_courte, manteau, gilet, jupe, short, pantalon, robe_courte, robe_longue, robe_sans_manches, jupe_longue, combinaison
- Graceful degradation: if ONNX file not found, `actif=False` — system works normally without it

---

## USB Autoinstall Deployment (Magasin)

ISO bootable Ubuntu Server 24.04 LTS qui installe toute la stack Fraude sur un PC magasin sans intervention. Tout est dans `deploy/usb-installer/`.

### Architecture du payload

L'ISO embarque l'image Docker **PRE-CONSTRUITE** (3.3 GB compressee) au lieu de la rebuild au boot du magasin. Avantage : pas besoin d'internet rapide pendant l'install (juste pour les paquets apt), et au 1er boot du laptop : `docker load` (~2 min) puis `compose up` (instantane). Les boots suivants : skip immediat via marker SHA256.

```
deploy/usb-installer/
├── build-iso.ps1                         # Script de build (Windows + Docker Desktop)
├── README-install.md                     # Doc utilisateur (install + maintenance)
├── autoinstall/
│   ├── user-data                         # cloud-init (autoinstall Ubuntu)
│   └── meta-data                         # vide (NoCloud datasource)
├── payload/                              # Tout ce qui sera copie dans /opt/fraude/
│   ├── .env                              # Config app (cameras vides → DB)
│   ├── docker-compose.yml                # Override avec image: au lieu de build:
│   ├── data/fraude.db                    # DB seedee (7 cameras, 2 actives)
│   ├── images/
│   │   ├── fraude-images.tar.gz          # Image Docker compressee (~3.3 GB)
│   │   └── fraude-images.image-id        # Cache pour skip docker save
│   ├── scripts/
│   │   ├── load-images.sh                # gunzip + docker load (idempotent via marker)
│   │   └── seed-fraude-db.py             # Genere data/fraude.db (7 cameras)
│   └── systemd/fraude.service            # Type=oneshot, ExecStartPre=load-images.sh
```

### Workflow complet

```powershell
# 1. Build l'image Docker en dev (genere fraude-fraud-detector:latest, ~6.8 GB)
docker compose build

# 2. (Une fois) Generer la DB seedee avec les 7 cameras
docker run --rm -v "${PWD}:/repo" -w /repo python:3.11-slim bash -c "pip install -q pydantic pydantic-settings && python deploy/usb-installer/payload/scripts/seed-fraude-db.py --output deploy/usb-installer/payload/data/fraude.db"

# 3. Build l'ISO (5 min docker save + 5 min xorriso = 10 min)
.\deploy\usb-installer\build-iso.ps1 -UbuntuIso "C:\Users\mhilali\Downloads\ubuntu-24.04.4-live-server-amd64.iso"
# Output : .\fraude-installer.iso (~6.5 GB)

# 4. Flasher avec Rufus (CRITIQUE : voir gotchas ci-dessous)
# 5. Boot le laptop magasin sur la cle, attendre install + reboot, dashboard accessible
```

### 7 Cameras seedees par defaut

`payload/scripts/seed-fraude-db.py` insere 7 cameras du magasin (RTSP Hikvision sur 192.168.1.5 et 192.168.1.3). **2 actives** (Caisse en mode caisse, CAM5 en mode vol), **5 inactives** (CAM1, CAM2, CAM4, CAM6, CAM8) — a activer une par une depuis le dashboard selon CPU disponible.

### Gotchas critiques

**1. Image Docker pre-construite vs rebuild au boot**
- Strategie initiale : `docker compose up -d --build` au 1er boot → echoue souvent (ADSL magasin lent, pip download 600 MB, build 30-60 min, systemd timeout)
- Strategie actuelle : `docker save` cote dev, `docker load` au 1er boot magasin (~2 min)
- Une SEULE image (`fraude-fraud-detector:latest`) sert pour les 2 services (`fraud-detector` utilise CMD par defaut, `fraud-dashboard` override avec `streamlit run`)
- `build-iso.ps1` utilise un cache via `payload/images/fraude-images.image-id` pour skip le `docker save` si l'image n'a pas change

**2. Rufus : OBLIGATOIRE GPT + UEFI (non CSM)**
- Si Rufus en mode MBR / Legacy, l'install se fait en BIOS mode, GRUB s'installe sur MBR au lieu de la partition EFI
- Au reboot, le BIOS UEFI ne trouve pas de bootloader sur le NVMe → loop "reset system" sur logo Lenovo
- **Toujours selectionner : Schema GPT + Systeme UEFI (non CSM) + Mode image DD**
- Verifier en F12 que l'entree USB commence par `UEFI:` (sinon legacy)

**3. Ethernet OBLIGATOIRE pendant l'install**
- L'autoinstall telecharge docker.io + cockpit + autres paquets depuis les repos Ubuntu (~200 MB)
- Sans internet : `curtin command system-install` retourne exit 100 sur docker.io → install fail
- WiFi non supporte par l'autoinstall actuel (pas de `wifis:` dans network config)
- Brancher cable RJ45 sur le meme reseau que les cameras

**4. BIOS T480s a verifier**
- Security → Secure Boot → **Disabled**
- Startup → UEFI/Legacy Boot → **UEFI Only**
- Apres install, F12 doit montrer une entree **"ubuntu"** (sinon GRUB EFI pas installe)

**5. Reboot post-install : debrancher la cle**
- L'autoinstall fait `shutdown: reboot` automatiquement
- Si la cle USB reste branchee ET que USB est en 1er dans boot order : loop d'install
- Solution simple : debrancher physiquement la cle pendant le reboot

**6. PowerShell 5.1 + Docker stderr**
- `docker image inspect` ecrit sur stderr quand l'image n'existe pas → avec `$ErrorActionPreference=Stop`, throw NativeCommandError
- Solution : wrapper dans try/catch (cf `build-iso.ps1` ligne 97-110)
- Pareil pour `docker save | docker run -i` : pipes binaires en PS 5.1 corrompent les donnees → utiliser `docker save -o file.tar` puis gzip via container Alpine

**7. Mot de passe par env var (PS 5.1 + Python crypt)**
- Python 3.11 emet un DeprecationWarning sur le module crypt
- En PS 5.1, ce warning sur stderr crash le script → utiliser `python -W ignore` + passer le password via env var (pas en string inline)

### load-images.sh : idempotence

Le script utilise `/var/lib/fraude/images-loaded` qui contient le SHA256 de l'archive `fraude-images.tar.gz`. Au 2eme boot et suivants, si le SHA matche → skip immediat. Si l'ISO est re-flashee avec une nouvelle image (SHA different) → re-load automatique. Pour forcer le re-load manuel : `sudo rm /var/lib/fraude/images-loaded && sudo systemctl restart fraude.service`.

### fraude.service systemd

- `Type=oneshot` + `RemainAfterExit=yes` (le service "termine" mais reste actif tant que les containers tournent)
- `ExecStartPre=/opt/fraude/scripts/load-images.sh` (load image, idempotent)
- `ExecStart=docker compose up -d --remove-orphans` (PAS de --build, PAS de --pull)
- `TimeoutStartSec=600` (5 min suffisent largement, vu qu'on ne build pas)
- `Restart=on-failure` + `RestartSec=30`

### Tunings post-install T480s (manuel via SSH apres install)

Non inclus dans l'autoinstall, a faire une fois le laptop installe et accessible :

```bash
ssh fraude@<ip>

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

# 3. Retention 7 jours (au lieu de 30 par defaut) : Dashboard > Administration > Metier
```

### Configuration WiFi T480s (apres install ethernet)

L'install autoinstall et le manuel se font en ethernet (cable RJ45 obligatoire pour apt). Une fois Ubuntu fonctionnel, on peut activer le WiFi pour deplacer le laptop sans cable.

**Identifier l'interface WiFi** :

```bash
ip link show | grep -E "wlp|wlan"
# Sur T480s : typiquement wlp61s0 (Intel AC 8265)
```

**Methode 1 : nmcli (le plus simple, recommande)** :

```bash
# Verifier si NetworkManager est installe
nmcli --version 2>/dev/null && echo "OK" || echo "A INSTALLER"

# Si pas installe :
sudo apt update && sudo apt install -y network-manager
sudo systemctl enable --now NetworkManager
echo "network: {config: disabled}" | sudo tee /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg

# Scanner les WiFi
sudo nmcli device wifi list

# Se connecter
sudo nmcli device wifi connect "TON_SSID" password "TON_MOT_DE_PASSE" ifname wlp61s0

# Verifier
nmcli connection show --active
ip a show wlp61s0
ping -c 3 -I wlp61s0 8.8.8.8
```

**Methode 2 : netplan (sans NetworkManager)** :

```bash
sudo apt install -y wpasupplicant
sudo nano /etc/netplan/60-wifi.yaml
```

Contenu (adapter `wlp61s0`, `TON_SSID`, `TON_MOT_DE_PASSE`) :
```yaml
network:
  version: 2
  wifis:
    wlp61s0:
      dhcp4: true
      optional: true
      access-points:
        "TON_SSID":
          password: "TON_MOT_DE_PASSE"
```

```bash
sudo chmod 600 /etc/netplan/60-wifi.yaml
sudo netplan apply
```

**⚠️ Important** : l'IP du laptop change quand on passe d'ethernet a WiFi (DHCP de la box WiFi != ethernet). Pour garder une IP fixe en WiFi : **configurer une reservation DHCP dans la box** sur la MAC adress WiFi du laptop. Sinon l'IP peut changer apres chaque reboot et il faut re-decouvrir l'IP via `nmap` / scan reseau.

### Acces remote via Tailscale

Pour acceder au laptop depuis l'exterieur du magasin (bureau, telephone, deplacement), on utilise **Tailscale** : VPN mesh peer-to-peer, gratuit (jusqu'a 100 devices), pas de port forwarding necessaire sur la box du magasin. **Avantage majeur** : l'IP Tailscale (100.x.x.x) est **fixe et persistante** — pas besoin de configurer une reservation DHCP sur la box WiFi.

**Setup (deja fait sur le T480s magasin) :**

```bash
# 1. Install
curl -fsSL https://tailscale.com/install.sh | sh

# 2. Auth (genere une URL a ouvrir dans navigateur pour login Google/MS/etc)
sudo tailscale up

# 3. Verifier
tailscale ip -4              # IP Tailscale (ex: 100.123.127.5)
tailscale status             # liste des autres devices du tailnet
hostname -f                  # nom complet (avec MagicDNS)
```

**Sur les machines admin (PC Windows, telephone, etc.) :**
- Telecharger : https://tailscale.com/download
- Installer + login avec **le meme compte** que le laptop magasin
- Tailscale t'attribue une IP 100.x.x.x — tu peux maintenant pinger le laptop magasin

**Acces aux services depuis n'importe quel device du tailnet :**
- Dashboard : http://`<tailscale-ip>`:8502
- Cockpit : https://`<tailscale-ip>`:9090
- MJPEG : http://`<tailscale-ip>`:8555/stream
- SSH : `ssh fraude@<tailscale-ip>`

**MagicDNS (optionnel mais pratique)** :
- Console : https://login.tailscale.com/admin/dns
- Toggle "Enable MagicDNS"
- Apres : utiliser `http://fraude:8502` au lieu de l'IP

**Pas besoin** : IP publique, port forwarding, DDNS, reservation DHCP, VPN OpenVPN, certificats SSL. Tout est gere par Tailscale en peer-to-peer (WireGuard sous le capot).

### Acces post-install

| Service | URL | Login |
|---------|-----|-------|
| Dashboard Streamlit | http://`<ip>`:8502 | (pas de login) |
| Cockpit (admin web) | https://`<ip>`:9090 | fraude / asx |
| MJPEG live camera | http://`<ip>`:8555/stream | (pas de login) |
| SSH | `ssh fraude` (alias `~/.ssh/config`) | cle ed25519 (fallback password : `asx`) |

---

## Package Installer (alternative a l'ISO autoinstall)

Si l'autoinstall echoue (BIOS bug, hardware specifique, etc.), on a une alternative : installer Ubuntu manuellement puis deployer un **package .tar.gz autonome** qui contient tout. Plus robuste, plus flexible.

### Workflow

```powershell
# 1. Cote dev (Windows + Docker Desktop)
docker compose build                                          # build image fraude-fraud-detector
.\deploy\package-installer\build-package.ps1                  # cree fraude-package.tar.gz (~3.4 GB)

# 2. Cote magasin (Linux Ubuntu deja installe)
scp fraude-package.tar.gz <user>@<ip>:~
ssh <user>@<ip>
tar xzf fraude-package.tar.gz && cd fraude-package
sudo ./install.sh                                             # install ~5-10 min
```

### Differences vs ISO autoinstall

| Aspect | ISO autoinstall | Package .tar.gz |
|--------|-----------------|-----------------|
| Linux | Installe automatiquement | A installer manuellement avant |
| Taille | 6.5 GB | 3.4 GB |
| Cas d'usage | Hardware standard, deploiement rapide | Hardware avec bug BIOS, machines deja en service |
| Reseau install | Ethernet (apt + paquets) | Ethernet (apt + paquets) |
| Image Docker | Embarquee dans /payload | Embarquee dans /images |

### Fichiers du package

```
deploy/package-installer/
├── build-package.ps1                    # Script Windows (genere le .tar.gz)
├── install.sh                           # Script Linux (a executer en sudo sur la machine cible)
└── README.md                            # Doc utilisateur
```

### install.sh : 8 etapes

1. apt install docker.io + docker-compose-v2 + cockpit + ufw + sqlite3 + curl
2. systemctl enable + start docker.service + cockpit.socket
3. usermod -aG docker $SUDO_USER
4. rsync payload → /opt/fraude + **chmod a+rwX sur data, recordings, snapshots, sounds** (critique : container UID != host UID)
5. cp systemd unit + systemctl enable fraude.service
6. ufw allow 22, 8502, 8555, 9090 + ufw enable
7. systemctl start fraude.service (load image + compose up)
8. Wait healthchecks (max 3 min)

### Bug critique : Phoenix BIOS T480s ne reconnait pas les entrees NVRAM Linux

Sur certains ThinkPad T480s avec **Phoenix SecureCore BIOS** (build 2018-2021, version <1.55), le BIOS **n'affiche pas les entrees NVRAM creees par Linux** dans son boot menu (F12). Resultat : meme si `grub-install` reussit et qu'`efibootmgr` cree une entree "ubuntu", le BIOS ne la voit pas et tente de booter "Windows Boot Manager" mort → reset loop.

**Symptomes :**
- Install Ubuntu reussit (autoinstall ou manuel)
- F12 ne montre que : NVMe0, Windows Boot Manager, PCI LAN (pas d'entree "ubuntu")
- Boot sur NVMe → reset loop infini sur logo Lenovo

**Verifications BIOS prealables (ne resolvent PAS le bug Phoenix) :**
- Security → Secure Boot → Disabled
- Startup → UEFI/Legacy Boot → UEFI Only
- Pas de CSM
- Cle USB d'install flashee en GPT (Rufus : Schema GPT, Systeme cible UEFI non CSM)

**Solution : "Microsoft Hack"** — on remplace `bootmgfw.efi` (Windows Boot Manager) par notre `shimx64.efi` (GRUB Ubuntu). Le BIOS, qui fait confiance aveugle a l'entree Microsoft, lance notre shim.

```bash
# Boot sur Ubuntu Desktop Live USB (PAS Server, on a besoin du GUI Live mode)
sudo -i

# Monter la partition EFI
mkdir -p /mnt/efi
mount /dev/nvme0n1p1 /mnt/efi

# 1. Copier nos fichiers Ubuntu sous le nom Microsoft
mkdir -p /mnt/efi/EFI/Microsoft/Boot
cp /mnt/efi/EFI/ubuntu/shimx64.efi  /mnt/efi/EFI/Microsoft/Boot/bootmgfw.efi
cp /mnt/efi/EFI/ubuntu/grubx64.efi  /mnt/efi/EFI/Microsoft/Boot/grubx64.efi
cp /mnt/efi/EFI/ubuntu/grub.cfg     /mnt/efi/EFI/Microsoft/Boot/grub.cfg
cp /mnt/efi/EFI/ubuntu/mmx64.efi    /mnt/efi/EFI/Microsoft/Boot/mmx64.efi

# 2. (Si on a un ancien Windows Boot Manager mort dans NVRAM, le supprimer)
efibootmgr -v        # repere son numero (souvent Boot0000)
efibootmgr -B -b 0000

# 3. Creer l'entree NVRAM "Windows Boot Manager" pointant sur notre fichier hacke
efibootmgr --create --disk /dev/nvme0n1 --part 1 --label "Windows Boot Manager" --loader "\EFI\Microsoft\Boot\bootmgfw.efi"

# 4. Verifier (BootOrder doit avoir notre nouvelle entree en 1ere position)
efibootmgr -v

# 5. Reboot
sync
systemctl reboot -i

# Au reboot : debrancher la cle USB
# Le BIOS lance bootmgfw.efi (qu'il croit etre Windows) → notre shim → GRUB → Ubuntu
```

**Pourquoi ca marche :** le BIOS Phoenix T480s **filtre le boot menu sur les labels/GUIDs reconnus** (Microsoft, Lenovo Diagnostics, USB CD, etc.). Le label "Windows Boot Manager" est dans cette whitelist. En creant une entree avec ce label exact pointant vers notre shim, on contourne le filtre.

**Alternative si ca ne marche pas :** mise a jour BIOS T480s vers 1.55+ (corrige plusieurs bugs UEFI). Telechargement : https://pcsupport.lenovo.com/products/laptops-and-netbooks/thinkpad-t-series-laptops/thinkpad-t480s

### Bug critique : permissions volumes Docker (DB readonly)

Le user `fraude` **dans le container** (UID 1001 typiquement, cree par `useradd -r` dans le Dockerfile) a un UID different du user `fraude` **sur le host** (UID 1000 typiquement, cree pendant l'install Ubuntu). Sans correction, le container ne peut pas ecrire la DB SQLite ni les recordings.

**Symptome :**
```
sqlite3.OperationalError: attempt to write a readonly database
fraud-detector  Restarting (1) X seconds ago
fraud-dashboard Up X minutes (unhealthy)
```

**Fix dans `install.sh` (etape 4)** : apres rsync, chmod a+rwX sur les volumes monteables :
```bash
chmod -R a+rwX /opt/fraude/data /opt/fraude/recordings /opt/fraude/snapshots /opt/fraude/sounds
```

**Fix manuel sur deploy existant :**
```bash
sudo chmod -R a+rwX /opt/fraude/data /opt/fraude/recordings /opt/fraude/snapshots /opt/fraude/sounds
sudo systemctl restart fraude.service
```

**Long terme (TODO)** : modifier le Dockerfile pour utiliser un UID matchant le host (1000 par defaut) :
```dockerfile
ARG USER_ID=1000
RUN groupadd -g ${USER_ID} fraude && useradd -u ${USER_ID} -g fraude -m -s /bin/bash fraude
```
Puis builder avec `docker build --build-arg USER_ID=$(id -u) ...`

### Apres install : groupe docker pas pris en compte

`install.sh` fait `usermod -aG docker $SUDO_USER`, mais la session SSH actuelle n'a pas le nouveau groupe (les groupes sont charges au login).

**Symptome :**
```
$ docker ps
permission denied while trying to connect to the Docker daemon socket
```

**Fix :**
```bash
newgrp docker     # active le groupe pour la session courante
# OU
exit              # logout SSH puis re-login
```
