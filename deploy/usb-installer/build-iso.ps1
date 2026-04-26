<#
.SYNOPSIS
    Construit une ISO Ubuntu Server 24.04 autoinstall avec l'app Fraude embarquee.

.DESCRIPTION
    Remasterize une ISO Ubuntu Server officielle pour qu'elle :
      - Boot directement en autoinstall (cloud-init / user-data)
      - Cree l'utilisateur 'fraude' avec mot de passe au choix (defaut: ASX@admin)
      - Installe Docker, Cockpit, SSH
      - Copie le payload Fraude dans /opt/fraude
      - Active le service systemd qui lance la stack au boot

    Le script utilise Docker pour faire le remasterisation (xorriso) cote Linux.
    Aucun WSL / Cygwin requis sur la machine de build.

.PARAMETER UbuntuIso
    Chemin vers l'ISO Ubuntu Server 24.04 LTS originale.
    Telecharger depuis : https://releases.ubuntu.com/24.04/

.PARAMETER OutputIso
    Chemin de l'ISO de sortie (defaut: .\fraude-installer.iso)

.PARAMETER Password
    Mot de passe en clair pour l'utilisateur fraude (defaut: ASX@admin).
    Il sera hashe (SHA-512 crypt) avant d'etre integre dans user-data.

.EXAMPLE
    .\build-iso.ps1 -UbuntuIso "C:\Downloads\ubuntu-24.04.1-live-server-amd64.iso"

.EXAMPLE
    .\build-iso.ps1 -UbuntuIso ".\ubuntu.iso" -OutputIso ".\out.iso" -Password "MonPass!"

.NOTES
    Pre-requis : Docker Desktop demarre.
    Duree typique : 3-5 minutes (selon connexion pour pull ubuntu:24.04).
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$UbuntuIso,

    [string]$OutputIso = ".\fraude-installer.iso",

    [string]$Password = "ASX@admin",

    # Image Docker a embarquer (doit etre construite avant : docker compose build a la racine du repo)
    [string]$DockerImage = "fraude-fraud-detector:latest",

    # Force le re-export de l'image meme si le cache est valide
    [switch]$ForceImageSave
)

$ErrorActionPreference = "Stop"

# ---- Resoudre les chemins en absolus ----
$ScriptDir = $PSScriptRoot
$RepoRoot  = Resolve-Path (Join-Path $ScriptDir "..\..")
$UbuntuIso = Resolve-Path $UbuntuIso
$OutputIso = Join-Path $PWD $OutputIso

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Build ISO Fraude Autoinstall" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Repo root  : $RepoRoot"
Write-Host "ISO source : $UbuntuIso"
Write-Host "ISO sortie : $OutputIso"
Write-Host ""

# ---- 1. Verifications ----
if (-not (Test-Path $UbuntuIso)) {
    Write-Error "ISO Ubuntu introuvable : $UbuntuIso"
}

try {
    docker version --format "{{.Server.Version}}" | Out-Null
} catch {
    Write-Error "Docker n'est pas demarre. Lance Docker Desktop puis re-execute."
}

# ---- 2. Verifier que les modeles ONNX sont presents ----
$ModelsDir = Join-Path $RepoRoot "models"
$RequiredModels = @("yolov8n.onnx", "yolov8n-pose.onnx", "yolov8n-oiv7.onnx")
foreach ($m in $RequiredModels) {
    $p = Join-Path $ModelsDir $m
    if (-not (Test-Path $p)) {
        Write-Error "Modele manquant : $p`nLance d'abord : docker compose build (a la racine du repo) pour les generer."
    }
}
Write-Host "[OK] Modeles ONNX presents" -ForegroundColor Green

# ---- 2.5. Verifier que l'image Docker Fraude existe + l'exporter en .tar.gz ----
# On embarque l'image PRE-CONSTRUITE dans la cle USB pour eviter au magasin
# de re-telecharger 600 MB de deps Python + reconstruire l'image au 1er boot
# (sur reseau lent ADSL : 30-60 min, parfois plus). Avec l'image embarquee :
# load + up = ~2 min au 1er boot, <30 sec aux boots suivants.

Write-Host "[..] Verification de l'image Docker $DockerImage..." -ForegroundColor Yellow
# try/catch necessaire : docker image inspect ecrit sur stderr si l'image n'existe pas,
# et avec $ErrorActionPreference=Stop, PowerShell 5.1 throw sur NativeCommandError.
$ImageInfo = $null
try {
    $ImageInfo = docker image inspect $DockerImage --format "{{.Id}}|{{.Size}}|{{.Created}}"
} catch {
    $ImageInfo = $null
}
if (-not $ImageInfo) {
    Write-Error @"
Image Docker introuvable : $DockerImage
Lance d'abord la construction depuis la racine du repo :
    cd "$RepoRoot"
    docker compose build
Puis re-execute ce script.
"@
}
$ImageParts   = $ImageInfo.Trim().Split("|")
$ImageId      = $ImageParts[0]
$ImageSizeMB  = [math]::Round([int64]$ImageParts[1] / 1MB, 0)
$ImageCreated = $ImageParts[2]
Write-Host "[OK] Image trouvee : $ImageId" -ForegroundColor Green
Write-Host "      Taille decompressee : $ImageSizeMB MB"
Write-Host "      Creee le            : $ImageCreated"

# Cache : on stocke l'ID de l'image dans un sidecar pour eviter de re-saver
# si l'image n'a pas change (un docker save | gzip d'une image 6.8 GB prend ~5 min).
$ImagesDir   = Join-Path $ScriptDir "payload\images"
$ImageTarGz  = Join-Path $ImagesDir "fraude-images.tar.gz"
$ImageMarker = Join-Path $ImagesDir "fraude-images.image-id"
if (-not (Test-Path $ImagesDir)) { New-Item -ItemType Directory -Path $ImagesDir | Out-Null }

$NeedSave = $true
if ((Test-Path $ImageTarGz) -and (Test-Path $ImageMarker) -and (-not $ForceImageSave)) {
    $CachedId = (Get-Content $ImageMarker -Raw).Trim()
    if ($CachedId -eq $ImageId) {
        $TarSizeMB = [math]::Round((Get-Item $ImageTarGz).Length / 1MB, 1)
        Write-Host "[OK] Cache valide : $ImageTarGz ($TarSizeMB MB) - skip docker save" -ForegroundColor Green
        $NeedSave = $false
    } else {
        Write-Host "[..] Image ID change ($CachedId -> $ImageId), re-export..." -ForegroundColor Yellow
    }
}

if ($NeedSave) {
    Write-Host "[..] Export de l'image $DockerImage (peut prendre 3-5 min)..." -ForegroundColor Yellow
    if (Test-Path $ImageTarGz) { Remove-Item -Force $ImageTarGz }

    # Strategie : docker save -o file.tar (binaire ecrit directement par dockerd, pas
    # via le pipe PowerShell qui corromprait les donnees binaires en PS 5.1), puis
    # gzip via un container Alpine qui a gzip natif (Windows n'a pas gzip et
    # PowerShell Compress-Archive ne fait que du .zip).
    $TempTar = Join-Path $env:TEMP "fraude-images-$([guid]::NewGuid().Guid).tar"
    try {
        Write-Host "      [1/2] docker save -> $TempTar..." -ForegroundColor Gray
        docker save $DockerImage -o $TempTar
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $TempTar)) {
            Write-Error "Echec docker save -o (exit code $LASTEXITCODE)"
        }
        $TarRawMB = [math]::Round((Get-Item $TempTar).Length / 1MB, 1)
        Write-Host "      [1/2] OK : $TarRawMB MB (non compresse)" -ForegroundColor Gray

        Write-Host "      [2/2] Compression gzip via container Alpine..." -ForegroundColor Gray
        $ImagesDirAbs = (Resolve-Path $ImagesDir).Path
        $TempTarAbs   = (Resolve-Path $TempTar).Path
        $env:MSYS_NO_PATHCONV = "1"
        docker run --rm `
            -v "${TempTarAbs}:/in.tar:ro" `
            -v "${ImagesDirAbs}:/out" `
            alpine:3.19 `
            sh -c "gzip -1 -c /in.tar > /out/fraude-images.tar.gz"
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $ImageTarGz)) {
            Write-Error "Echec compression gzip (exit code $LASTEXITCODE)"
        }
    } finally {
        if (Test-Path $TempTar) { Remove-Item -Force $TempTar }
    }

    $TarSizeMB = [math]::Round((Get-Item $ImageTarGz).Length / 1MB, 1)
    Write-Host "[OK] Image exportee : $TarSizeMB MB compresses" -ForegroundColor Green
    $ImageId | Out-File -FilePath $ImageMarker -Encoding ascii -NoNewline
}

# ---- 3. Generer le hash du mot de passe ----
# On passe le password via env var (pas de soucis d'echappement) et -W ignore
# pour silencer le DeprecationWarning de Python 3.11 sur le module crypt (qui
# polluerait stderr et casserait PowerShell 5.1 avec ErrorAction=Stop).
Write-Host "[..] Generation du hash SHA-512 pour le mot de passe..." -ForegroundColor Yellow
$PasswordHash = docker run --rm -e FRAUDE_PWD="$Password" python:3.11-slim `
    python -W ignore -c "import os, crypt; print(crypt.crypt(os.environ['FRAUDE_PWD'], crypt.mksalt(crypt.METHOD_SHA512)))"
if ([string]::IsNullOrWhiteSpace($PasswordHash)) {
    Write-Error "Echec generation du hash de mot de passe."
}
$PasswordHash = $PasswordHash.Trim()
Write-Host "[OK] Hash genere" -ForegroundColor Green

# ---- 4. Preparer le repertoire de staging ----
$Staging = Join-Path $env:TEMP "fraude-iso-build"
if (Test-Path $Staging) { Remove-Item -Recurse -Force $Staging }
New-Item -ItemType Directory -Path $Staging | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Staging "autoinstall") | Out-Null
New-Item -ItemType Directory -Path (Join-Path $Staging "payload")     | Out-Null

# ---- 5. user-data (avec hash injecte) + meta-data ----
$UserDataTpl = Get-Content (Join-Path $ScriptDir "autoinstall\user-data") -Raw
$UserData    = $UserDataTpl.Replace("__PASSWORD_HASH__", $PasswordHash)
$UserData | Out-File -FilePath (Join-Path $Staging "autoinstall\user-data") -Encoding ascii -NoNewline
Copy-Item (Join-Path $ScriptDir "autoinstall\meta-data") (Join-Path $Staging "autoinstall\meta-data")
Write-Host "[OK] user-data prepare avec hash injecte" -ForegroundColor Green

# ---- 6. Construction du payload (code + modeles + DB seed + sounds + .env) ----
$PayloadDst = Join-Path $Staging "payload"
Write-Host "[..] Copie du payload..." -ForegroundColor Yellow

# Code applicatif (utile pour debug / hotfix sur le terrain ; le container
# l'embarque deja, mais l'avoir sur le host facilite l'inspection via SSH)
Copy-Item -Recurse (Join-Path $RepoRoot "app")        $PayloadDst
Copy-Item -Recurse (Join-Path $RepoRoot "dashboard")  $PayloadDst
Copy-Item -Recurse (Join-Path $RepoRoot "scripts")    (Join-Path $PayloadDst "scripts-app")
Copy-Item -Recurse (Join-Path $RepoRoot "sounds")     $PayloadDst
# Dockerfile + requirements.txt copies pour rebuild d'urgence (sans cle USB)
Copy-Item          (Join-Path $RepoRoot "Dockerfile") $PayloadDst
Copy-Item          (Join-Path $RepoRoot "requirements.txt")   $PayloadDst

# docker-compose.yml : on prend la version "magasin" (image: au lieu de build:),
# PAS celle du repo (qui forcerait un rebuild a chaque demarrage du service)
Copy-Item (Join-Path $ScriptDir "payload\docker-compose.yml") (Join-Path $PayloadDst "docker-compose.yml")

# Modeles ONNX (montes en volume, masquent ceux du container)
Copy-Item -Recurse $ModelsDir (Join-Path $PayloadDst "models")

# Image Docker pre-construite (3-4 GB compressee)
Copy-Item -Recurse $ImagesDir (Join-Path $PayloadDst "images")

# Sources autoinstall : .env, systemd unit, scripts d'install, DB seedee
Copy-Item (Join-Path $ScriptDir "payload\.env") (Join-Path $PayloadDst ".env")
Copy-Item -Recurse (Join-Path $ScriptDir "payload\systemd") $PayloadDst
Copy-Item -Recurse (Join-Path $ScriptDir "payload\scripts") $PayloadDst
New-Item -ItemType Directory -Path (Join-Path $PayloadDst "data") | Out-Null
Copy-Item (Join-Path $ScriptDir "payload\data\fraude.db") (Join-Path $PayloadDst "data\fraude.db")

# Repertoires runtime vides (pour eviter chown errors apres install)
foreach ($d in @("recordings", "snapshots")) {
    New-Item -ItemType Directory -Path (Join-Path $PayloadDst $d) | Out-Null
    "" | Out-File -FilePath (Join-Path $PayloadDst "$d\.gitkeep") -Encoding ascii -NoNewline
}

$payloadSize = [math]::Round((Get-ChildItem -Recurse $PayloadDst | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
Write-Host "[OK] Payload prepare ($payloadSize MB)" -ForegroundColor Green

# ---- 7. Lancer le remasterisation dans un container Linux ----
Write-Host "[..] Lancement du container builder (xorriso)..." -ForegroundColor Yellow

# On monte :
#   /staging   = autoinstall/ + payload/ pretes
#   /iso-src   = ISO Ubuntu originale (read-only, en lecture)
#   /iso-out   = repertoire de sortie pour la nouvelle ISO
$IsoDir   = Split-Path $UbuntuIso -Parent
$IsoFile  = Split-Path $UbuntuIso -Leaf
$OutDir   = Split-Path $OutputIso -Parent
$OutFile  = Split-Path $OutputIso -Leaf

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

# Script bash execute dans le container
# Approche: modification "in place" de l'ISO Ubuntu avec xorriso -boot_image any replay
# qui PRESERVE automatiquement la structure de boot (BIOS + UEFI hybride). On extrait
# uniquement les fichiers a modifier (grub.cfg + isolinux), on les patche, puis on les
# remappe sur l'ISO d'origine. Pas besoin de reconstruire toutes les options de boot.
$BuilderScript = @'
set -euo pipefail
set -x
export DEBIAN_FRONTEND=noninteractive

echo "[builder] apt install xorriso + sed..."
apt-get update -qq
apt-get install -y --no-install-recommends xorriso sed coreutils

ISO_SRC="/iso-src/__ISO_FILE__"
ISO_OUT="/iso-out/__OUT_FILE__"

[ -f "$ISO_SRC" ] || { echo "ERREUR: ISO source introuvable: $ISO_SRC"; ls -la /iso-src; exit 1; }
echo "[builder] ISO source OK: $(ls -lh $ISO_SRC | awk '{print $5}')"

# ---- 1. Preparer l'overlay (uniquement les fichiers qui changent) ----
OVERLAY=/tmp/overlay
rm -rf $OVERLAY
mkdir -p $OVERLAY/boot/grub $OVERLAY/autoinstall $OVERLAY/payload

# Copier autoinstall + payload depuis le staging
cp -a /staging/autoinstall/. $OVERLAY/autoinstall/
cp -a /staging/payload/.     $OVERLAY/payload/

# Extraire uniquement grub.cfg pour le modifier
xorriso -osirrox on -indev "$ISO_SRC" \
    -extract /boot/grub/grub.cfg $OVERLAY/boot/grub/grub.cfg
chmod +w $OVERLAY/boot/grub/grub.cfg

# Verifier qu'on a bien le grub.cfg
[ -f "$OVERLAY/boot/grub/grub.cfg" ] || { echo "ERREUR: grub.cfg pas extrait"; exit 1; }
echo "[builder] grub.cfg avant patch:"
grep -E "(timeout|linux|---)" $OVERLAY/boot/grub/grub.cfg | head -10

# ---- 2. Patcher grub.cfg ----
# a) Ajouter "autoinstall ds=nocloud\;s=/cdrom/autoinstall/" avant " ---"
#    Note: on echappe le ; pour grub avec \;
sed -i 's| ---| autoinstall ds=nocloud\\;s=/cdrom/autoinstall/ ---|g' \
    $OVERLAY/boot/grub/grub.cfg
# b) Reduire le timeout (souvent "set timeout=30" ou "set timeout_style=...")
sed -i 's/set timeout=[0-9]*/set timeout=1/g' $OVERLAY/boot/grub/grub.cfg

echo "[builder] grub.cfg apres patch:"
grep -E "(timeout|autoinstall)" $OVERLAY/boot/grub/grub.cfg | head -10

# ---- 3. Patcher isolinux/txt.cfg si present (BIOS legacy) ----
if xorriso -osirrox on -indev "$ISO_SRC" -extract /isolinux/txt.cfg /tmp/txt.cfg 2>/dev/null; then
    mkdir -p $OVERLAY/isolinux
    chmod +w /tmp/txt.cfg
    sed -i 's| ---| autoinstall ds=nocloud;s=/cdrom/autoinstall/ ---|g' /tmp/txt.cfg
    cp /tmp/txt.cfg $OVERLAY/isolinux/txt.cfg
    echo "[builder] isolinux/txt.cfg patche"
else
    echo "[builder] Pas d'isolinux/txt.cfg (Ubuntu 24.04 n'en a souvent plus, OK)"
fi

# ---- 4. Repackager : remap des fichiers modifies sur l'ISO d'origine ----
# -boot_image any replay : preserve toute la structure de boot (UEFI + BIOS hybride)
# -map src dst           : ajoute/remplace fichiers/dossiers dans l'ISO de sortie
echo "[builder] Repackaging ISO avec xorriso (boot replay)..."
rm -f "$ISO_OUT"

XORRISO_ARGS=(
    -indev "$ISO_SRC"
    -outdev "$ISO_OUT"
    -boot_image any replay
    -volid "Fraude Auto"
    -compliance no_emul_toc
    -map $OVERLAY/boot/grub/grub.cfg /boot/grub/grub.cfg
    -map $OVERLAY/autoinstall        /autoinstall
    -map $OVERLAY/payload            /payload
)
if [ -f $OVERLAY/isolinux/txt.cfg ]; then
    XORRISO_ARGS+=(-map $OVERLAY/isolinux/txt.cfg /isolinux/txt.cfg)
fi

xorriso "${XORRISO_ARGS[@]}"

echo "[builder] Verification de l'ISO produite..."
ls -lh "$ISO_OUT"
echo "[builder] OK"
'@

# Substituer les noms de fichiers
$BuilderScript = $BuilderScript.Replace("__ISO_FILE__", $IsoFile).Replace("__OUT_FILE__", $OutFile)

# Ecrire le script dans le staging
$BuilderScript | Out-File -FilePath (Join-Path $Staging "builder.sh") -Encoding ascii -NoNewline

# Lancer Docker
$env:MSYS_NO_PATHCONV = "1"
docker run --rm `
    -v "${Staging}:/staging" `
    -v "${IsoDir}:/iso-src:ro" `
    -v "${OutDir}:/iso-out" `
    ubuntu:24.04 `
    bash /staging/builder.sh

if ($LASTEXITCODE -ne 0) {
    Write-Error "Echec du build ISO (xorriso). Voir les logs ci-dessus."
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "[OK] ISO construite : $OutputIso" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
$isoSize = [math]::Round((Get-Item $OutputIso).Length / 1MB, 1)
Write-Host "Taille : $isoSize MB"
Write-Host ""
Write-Host "Prochaines etapes :" -ForegroundColor Cyan
Write-Host "  1. Flasher l'ISO sur la cle USB (32 GB) avec Rufus"
Write-Host "       https://rufus.ie/  -  selectionner 'Mode image DD'"
Write-Host "  2. Brancher la cle sur la machine cible et booter dessus"
Write-Host "  3. Attendre ~10-15 min : install + reboot automatiques"
Write-Host "  4. Apres reboot, le service fraude.service demarre tout seul"
Write-Host "       Dashboard  : http://<ip-machine>:8502"
Write-Host "       Cockpit    : https://<ip-machine>:9090   (login: fraude / $Password)"
Write-Host "       SSH        : ssh fraude@<ip-machine>"
Write-Host ""
