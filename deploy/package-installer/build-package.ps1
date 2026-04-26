<#
.SYNOPSIS
    Construit un package .tar.gz de Fraude pour deploiement sur Linux Ubuntu deja installe.

.DESCRIPTION
    Cree fraude-package.tar.gz contenant :
      - install.sh + README.md (script d'install + doc utilisateur)
      - Image Docker pre-construite (fraude-images.tar.gz, ~3.3 GB compressee)
      - Code app + dashboard + scripts (pour debug/hotfix)
      - Modeles ONNX (montes en volume au runtime)
      - DB SQLite seedee avec 7 cameras du magasin
      - docker-compose.yml override (image: au lieu de build:)
      - systemd unit fraude.service
      - .env de config

    Utilise Docker Desktop pour :
      - docker save (export de l'image fraude-fraud-detector:latest)
      - tar + gzip via container Alpine (Windows n'a pas tar/gzip natifs)

.PARAMETER OutputPackage
    Chemin du .tar.gz de sortie (defaut: .\fraude-package.tar.gz)

.PARAMETER DockerImage
    Image Docker a embarquer (defaut: fraude-fraud-detector:latest).
    Doit etre construite avant : `docker compose build` a la racine du repo.

.PARAMETER ForceImageSave
    Force le re-export de l'image meme si le cache est valide.

.EXAMPLE
    .\build-package.ps1
    # Output : .\fraude-package.tar.gz (~3.4 GB)

.EXAMPLE
    .\build-package.ps1 -OutputPackage "D:\fraude-package-v1.tar.gz" -ForceImageSave

.NOTES
    Pre-requis : Docker Desktop demarre + image fraude-fraud-detector:latest construite
    Duree typique : 5-10 minutes (selon si l'image doit etre re-saved)
#>

param(
    [string]$OutputPackage = ".\fraude-package.tar.gz",
    [string]$DockerImage = "fraude-fraud-detector:latest",
    [switch]$ForceImageSave
)

$ErrorActionPreference = "Stop"

# ---- Resoudre les chemins ----
$ScriptDir     = $PSScriptRoot
$RepoRoot      = Resolve-Path (Join-Path $ScriptDir "..\..")
$OutputPackage = Join-Path $PWD $OutputPackage

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Build Package Fraude (.tar.gz)" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Repo root      : $RepoRoot"
Write-Host "Image Docker   : $DockerImage"
Write-Host "Output package : $OutputPackage"
Write-Host ""

# ---- 1. Verifications ----
try {
    docker version --format "{{.Server.Version}}" | Out-Null
} catch {
    Write-Error "Docker n'est pas demarre. Lance Docker Desktop puis re-execute."
}

# Modeles ONNX
$ModelsDir = Join-Path $RepoRoot "models"
$RequiredModels = @("yolov8n.onnx", "yolov8n-pose.onnx", "yolov8n-oiv7.onnx")
foreach ($m in $RequiredModels) {
    if (-not (Test-Path (Join-Path $ModelsDir $m))) {
        Write-Error "Modele manquant : $(Join-Path $ModelsDir $m)`nLance d'abord : docker compose build"
    }
}
Write-Host "[OK] Modeles ONNX presents" -ForegroundColor Green

# DB seedee
$SeededDB = Join-Path $RepoRoot "deploy\usb-installer\payload\data\fraude.db"
if (-not (Test-Path $SeededDB)) {
    Write-Error @"
DB seedee manquante : $SeededDB
Genere-la d'abord :
    docker run --rm -v "${RepoRoot}:/repo" -w /repo python:3.11-slim bash -c "pip install -q pydantic pydantic-settings && python deploy/usb-installer/payload/scripts/seed-fraude-db.py --output deploy/usb-installer/payload/data/fraude.db"
"@
}
Write-Host "[OK] DB seedee presente ($([math]::Round((Get-Item $SeededDB).Length / 1KB, 1)) KB)" -ForegroundColor Green

# Image Docker
Write-Host "[..] Verification de l'image Docker $DockerImage..." -ForegroundColor Yellow
$ImageInfo = $null
try {
    $ImageInfo = docker image inspect $DockerImage --format "{{.Id}}|{{.Size}}"
} catch {
    $ImageInfo = $null
}
if (-not $ImageInfo) {
    Write-Error @"
Image Docker introuvable : $DockerImage
Lance d'abord :
    cd "$RepoRoot"
    docker compose build
"@
}
$ImageParts  = $ImageInfo.Trim().Split("|")
$ImageId     = $ImageParts[0]
$ImageSizeMB = [math]::Round([int64]$ImageParts[1] / 1MB, 0)
Write-Host "[OK] Image trouvee : $ImageId ($ImageSizeMB MB decompresse)" -ForegroundColor Green

# ---- 2. Export de l'image (avec cache) ----
# On reutilise le cache du build-iso.ps1 (memes fichiers dans deploy/usb-installer/payload/images/)
$ImagesDir   = Join-Path $RepoRoot "deploy\usb-installer\payload\images"
$ImageTarGz  = Join-Path $ImagesDir "fraude-images.tar.gz"
$ImageMarker = Join-Path $ImagesDir "fraude-images.image-id"
if (-not (Test-Path $ImagesDir)) { New-Item -ItemType Directory -Path $ImagesDir | Out-Null }

$NeedSave = $true
if ((Test-Path $ImageTarGz) -and (Test-Path $ImageMarker) -and (-not $ForceImageSave)) {
    $CachedId = (Get-Content $ImageMarker -Raw).Trim()
    if ($CachedId -eq $ImageId) {
        $TarSizeMB = [math]::Round((Get-Item $ImageTarGz).Length / 1MB, 1)
        Write-Host "[OK] Cache image valide ($TarSizeMB MB) - skip docker save" -ForegroundColor Green
        $NeedSave = $false
    }
}

if ($NeedSave) {
    Write-Host "[..] Export de l'image (3-5 min)..." -ForegroundColor Yellow
    if (Test-Path $ImageTarGz) { Remove-Item -Force $ImageTarGz }

    $TempTar = Join-Path $env:TEMP "fraude-images-$([guid]::NewGuid().Guid).tar"
    try {
        Write-Host "      [1/2] docker save -> $TempTar" -ForegroundColor Gray
        docker save $DockerImage -o $TempTar
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $TempTar)) {
            Write-Error "Echec docker save (exit $LASTEXITCODE)"
        }

        Write-Host "      [2/2] gzip via container Alpine" -ForegroundColor Gray
        $ImagesDirAbs = (Resolve-Path $ImagesDir).Path
        $TempTarAbs   = (Resolve-Path $TempTar).Path
        $env:MSYS_NO_PATHCONV = "1"
        docker run --rm `
            -v "${TempTarAbs}:/in.tar:ro" `
            -v "${ImagesDirAbs}:/out" `
            alpine:3.19 `
            sh -c "gzip -1 -c /in.tar > /out/fraude-images.tar.gz"
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $ImageTarGz)) {
            Write-Error "Echec compression gzip (exit $LASTEXITCODE)"
        }
    } finally {
        if (Test-Path $TempTar) { Remove-Item -Force $TempTar }
    }

    $TarSizeMB = [math]::Round((Get-Item $ImageTarGz).Length / 1MB, 1)
    Write-Host "[OK] Image exportee : $TarSizeMB MB compresses" -ForegroundColor Green
    $ImageId | Out-File -FilePath $ImageMarker -Encoding ascii -NoNewline
}

# ---- 3. Preparation du staging ----
$Staging = Join-Path $env:TEMP "fraude-package-build"
if (Test-Path $Staging) { Remove-Item -Recurse -Force $Staging }
New-Item -ItemType Directory -Path $Staging | Out-Null

$PackageDir = Join-Path $Staging "fraude-package"
New-Item -ItemType Directory -Path $PackageDir | Out-Null

Write-Host "[..] Preparation du staging dans $PackageDir..." -ForegroundColor Yellow

# Code applicatif
Copy-Item -Recurse (Join-Path $RepoRoot "app")        $PackageDir
Copy-Item -Recurse (Join-Path $RepoRoot "dashboard")  $PackageDir
Copy-Item -Recurse (Join-Path $RepoRoot "scripts")    (Join-Path $PackageDir "scripts-app")
Copy-Item -Recurse (Join-Path $RepoRoot "sounds")     $PackageDir
Copy-Item          (Join-Path $RepoRoot "Dockerfile") $PackageDir
Copy-Item          (Join-Path $RepoRoot "requirements.txt") $PackageDir

# docker-compose.yml : la version "magasin" (image: au lieu de build:)
Copy-Item (Join-Path $RepoRoot "deploy\usb-installer\payload\docker-compose.yml") (Join-Path $PackageDir "docker-compose.yml")

# Modeles ONNX
Copy-Item -Recurse $ModelsDir (Join-Path $PackageDir "models")

# Image Docker (3.3 GB)
Copy-Item -Recurse $ImagesDir (Join-Path $PackageDir "images")
# On nettoie le marker du cache (pas utile pour l'utilisateur final)
$MarkerInPkg = Join-Path $PackageDir "images\fraude-images.image-id"
if (Test-Path $MarkerInPkg) { Remove-Item -Force $MarkerInPkg }

# .env
Copy-Item (Join-Path $RepoRoot "deploy\usb-installer\payload\.env") (Join-Path $PackageDir ".env")

# systemd unit
Copy-Item -Recurse (Join-Path $RepoRoot "deploy\usb-installer\payload\systemd") $PackageDir

# scripts (load-images.sh)
Copy-Item -Recurse (Join-Path $RepoRoot "deploy\usb-installer\payload\scripts") $PackageDir

# DB seedee
New-Item -ItemType Directory -Path (Join-Path $PackageDir "data") | Out-Null
Copy-Item $SeededDB (Join-Path $PackageDir "data\fraude.db")

# Repertoires runtime vides (pour eviter chown errors)
foreach ($d in @("recordings", "snapshots")) {
    New-Item -ItemType Directory -Path (Join-Path $PackageDir $d) | Out-Null
    "" | Out-File -FilePath (Join-Path $PackageDir "$d\.gitkeep") -Encoding ascii -NoNewline
}

# install.sh + README.md (les fichiers cles du package)
Copy-Item (Join-Path $ScriptDir "install.sh") (Join-Path $PackageDir "install.sh")
Copy-Item (Join-Path $ScriptDir "README.md")  (Join-Path $PackageDir "README.md")

$pkgSize = [math]::Round((Get-ChildItem -Recurse $PackageDir | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
Write-Host "[OK] Staging prepare ($pkgSize MB non compresse)" -ForegroundColor Green

# ---- 4. tar.gz le tout via container Alpine ----
# (Windows n'a pas tar natif fiable + gzip + line endings)
Write-Host "[..] Compression tar.gz via container Alpine..." -ForegroundColor Yellow

$OutputDir  = Split-Path $OutputPackage -Parent
$OutputName = Split-Path $OutputPackage -Leaf
if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir | Out-Null }
$OutputDirAbs = (Resolve-Path $OutputDir).Path
$StagingAbs   = (Resolve-Path $Staging).Path

# Important : on tar depuis $StagingAbs pour avoir "fraude-package/" comme racine
# dans l'archive (et pas un chemin Windows). Et on fait chmod +x sur install.sh
# et load-images.sh AVANT le tar pour qu'ils restent executables sur Linux.
# Note : on copie vers /tmp d'abord pour pouvoir chmod (le mount /staging:ro
# est read-only), puis on tar depuis /tmp. Pas besoin de --transform car le
# dossier garde son nom (busybox tar ne supporte pas --transform de toute facon).
$env:MSYS_NO_PATHCONV = "1"
docker run --rm `
    -v "${StagingAbs}:/staging:ro" `
    -v "${OutputDirAbs}:/out" `
    alpine:3.19 `
    sh -c "cp -a /staging/fraude-package /tmp/fraude-package && chmod +x /tmp/fraude-package/install.sh /tmp/fraude-package/scripts/load-images.sh && tar czf /out/$OutputName -C /tmp fraude-package"

if ($LASTEXITCODE -ne 0 -or -not (Test-Path $OutputPackage)) {
    Write-Error "Echec creation du tar.gz (exit $LASTEXITCODE)"
}

# ---- 5. Cleanup staging ----
Remove-Item -Recurse -Force $Staging

# ---- 6. Resume ----
$pkgSizeMB = [math]::Round((Get-Item $OutputPackage).Length / 1MB, 1)
Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "[OK] Package construit : $OutputPackage" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host "Taille : $pkgSizeMB MB"
Write-Host ""
Write-Host "Deploiement sur la machine cible :" -ForegroundColor Cyan
Write-Host ""
Write-Host "  # 1. Copier le package (depuis ce PC Windows) :"
Write-Host "  scp $OutputPackage <user>@<ip-laptop>:~/"
Write-Host ""
Write-Host "  # 2. Sur le laptop Linux :"
Write-Host "  tar xzf $(Split-Path $OutputPackage -Leaf)"
Write-Host "  cd fraude-package"
Write-Host "  sudo ./install.sh"
Write-Host ""
Write-Host "  # 3. Acces apres install :"
Write-Host "  http://<ip>:8502   (Dashboard)"
Write-Host "  https://<ip>:9090  (Cockpit admin)"
Write-Host ""
