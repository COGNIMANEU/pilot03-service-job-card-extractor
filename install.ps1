# Job Card Extractor - Windows Installation Script
# Run with:
#   irm https://raw.githubusercontent.com/COGNIMANEU/pilot03-service-job-card-extractor/main/install.ps1 | iex

$ErrorActionPreference = "Stop"

$REPO_URL = "https://github.com/COGNIMANEU/pilot03-service-job-card-extractor.git"

function Write-Info { param($m) Write-Host "[INFO]  $m" -ForegroundColor Cyan }
function Write-Ok { param($m) Write-Host "[ OK ]  $m" -ForegroundColor Green }
function Write-Warn { param($m) Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Write-Err { param($m) Write-Host "[ERR]   $m" -ForegroundColor Red; exit 1 }

function Get-PythonVersion {
    try {
        $v = python --version 2>&1
        if ($v -match "Python (\d+\.\d+)") { return $matches[1] }
    }
    catch { }
    try {
        $v = python3 --version 2>&1
        if ($v -match "Python (\d+\.\d+)") { return $matches[1] }
    }
    catch { }
    return $null
}

Write-Info "Installing Job Card Extractor..."

# Check Python
$pythonVersion = Get-PythonVersion
if (-not $pythonVersion) {
    Write-Err "Python 3.6+ not found. Install from https://www.python.org/downloads/"
}
Write-Info "Python $pythonVersion found"
$pythonCmd = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "python3" }

# When piped via iex there is no script directory. Reuse the current checkout,
# or clone into a predictable child of the caller's current directory.
if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "job_card_extractor.py"))) {
    $checkout = $PSScriptRoot
}
elseif (Test-Path (Join-Path (Get-Location) "job_card_extractor.py")) {
    $checkout = (Get-Location).Path
}
else {
    $checkout = Join-Path (Get-Location) "pilot03-service-job-card-extractor"
    if (-not (Test-Path $checkout)) {
        git clone $REPO_URL $checkout
        if ($LASTEXITCODE -ne 0) { Write-Err "Failed to clone the extractor" }
    }
}
if (-not (Test-Path (Join-Path $checkout "job_card_extractor.py")) -or
    -not (Test-Path (Join-Path $checkout "requirements.txt"))) {
    Write-Err "Extractor checkout incomplete at $checkout"
}

$venvPath = Join-Path $checkout "venv"
if (-not (Test-Path $venvPath)) {
    Write-Info "Creating virtual environment at $venvPath"
    & $pythonCmd -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { Write-Err "Failed to create virtual environment" }
}
$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) { Write-Err "Virtual environment missing Python at $venvPython" }

Write-Info "Upgrading pip..."
& $venvPython -m pip install --upgrade pip | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Err "Failed to upgrade pip" }

Write-Info "Installing Python packages..."
& $venvPython -m pip install -r (Join-Path $checkout "requirements.txt") | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Err "Failed to install Python dependencies" }

Write-Ok "Installation complete!"
Write-Host ""
Write-Host "To activate the virtual environment, run:"
Write-Host "  $venvPath\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then run:"
Write-Host "  python $checkout\job_card_extractor.py <input.pdf> -o <output_dir>"