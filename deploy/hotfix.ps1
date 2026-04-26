<#
.SYNOPSIS
    Deploie un correctif sur le magasin sans rebuild d'image.

.DESCRIPTION
    Workflow git-based : commit/push local -> git pull magasin -> docker restart.
    Necessite : ssh fraude (alias dans ~/.ssh/config) + clone /home/fraude/fraude-src/.

.PARAMETER Service
    Quel service redemarrer apres le pull : detector, dashboard, all (defaut: all).

.PARAMETER NoRestart
    Pull uniquement, sans redemarrer (utile si tu veux verifier d'abord).

.EXAMPLE
    .\deploy\hotfix.ps1
    Push + pull + restart les 2 containers.

.EXAMPLE
    .\deploy\hotfix.ps1 -Service detector
    Push + pull + restart fraud-detector uniquement.

.EXAMPLE
    .\deploy\hotfix.ps1 -NoRestart
    Push + pull, sans toucher aux containers (validation manuelle ensuite).
#>
[CmdletBinding()]
param(
    [ValidateSet('detector', 'dashboard', 'all')]
    [string]$Service = 'all',
    [switch]$NoRestart
)

$ErrorActionPreference = 'Stop'

# Move to repo root (script is in deploy/, repo is parent)
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "=== Hotfix deploy ===" -ForegroundColor Cyan
Write-Host "Repo : $repoRoot"

# 1. Verifier qu'on a un working tree propre (sinon refus)
$dirty = git status --porcelain
if ($dirty) {
    Write-Host ""
    Write-Host "Working tree contient des modifs non commitees :" -ForegroundColor Yellow
    git status --short
    Write-Host ""
    $resp = Read-Host "Commiter ces modifs maintenant ? (y/N)"
    if ($resp -ne 'y') {
        Write-Host "Abandon. Commit ou stash tes changements puis relance." -ForegroundColor Red
        exit 1
    }
    git add -A
    $msg = Read-Host "Message du commit"
    if (-not $msg) { $msg = "hotfix: " + (Get-Date -Format "yyyy-MM-dd HH:mm") }
    git commit -m $msg
    if ($LASTEXITCODE -ne 0) { Write-Host "Commit echoue" -ForegroundColor Red; exit 1 }
}

# 2. Push (no-op si deja a jour)
Write-Host ""
Write-Host "--- git push ---" -ForegroundColor Cyan
git push
if ($LASTEXITCODE -ne 0) { Write-Host "Push echoue" -ForegroundColor Red; exit 1 }

# 3. Pull cote magasin + afficher les commits appliques
Write-Host ""
Write-Host "--- git pull cote magasin ---" -ForegroundColor Cyan
$pullOut = ssh fraude "cd ~/fraude-src && git fetch && echo '--- New commits:' && git log --oneline HEAD..origin/main && echo '--- Pulling:' && git pull --ff-only" 2>&1
$pullOut | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -ne 0) { Write-Host "Pull echoue" -ForegroundColor Red; exit 1 }

if ($NoRestart) {
    Write-Host ""
    Write-Host "Pull termine. Pas de restart demande (-NoRestart)." -ForegroundColor Green
    exit 0
}

# 4. Restart le ou les services
Write-Host ""
Write-Host "--- docker compose restart ($Service) ---" -ForegroundColor Cyan
$svcMap = @{
    'detector'  = 'fraud-detector'
    'dashboard' = 'dashboard'
    'all'       = ''
}
$svcName = $svcMap[$Service]
$restartCmd = if ($svcName) { "docker compose restart $svcName" } else { "docker compose restart" }
ssh fraude "cd /opt/fraude && $restartCmd" 2>&1 | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -ne 0) { Write-Host "Restart echoue" -ForegroundColor Red; exit 1 }

# 5. Wait + verifier health
Write-Host ""
Write-Host "--- Attente healthcheck (max 60s) ---" -ForegroundColor Cyan
$deadline = (Get-Date).AddSeconds(60)
$lastStatus = ""
while ((Get-Date) -lt $deadline) {
    $lastStatus = ssh fraude "docker ps --format '{{.Names}}|{{.Status}}'" 2>&1
    if ($lastStatus -match 'unhealthy') {
        Write-Host "Container unhealthy detecte :" -ForegroundColor Red
        $lastStatus | ForEach-Object { Write-Host $_ }
        exit 1
    }
    if ($lastStatus -notmatch 'starting') {
        break
    }
    Start-Sleep -Seconds 3
}
Write-Host ""
Write-Host "--- Etat final ---" -ForegroundColor Cyan
ssh fraude "docker ps --format 'table {{.Names}}\t{{.Status}}'" 2>&1 | ForEach-Object { Write-Host $_ }

Write-Host ""
Write-Host "Hotfix deploye." -ForegroundColor Green
