# Job Card Extractor - Windows Installation Script
# Run with:
#   irm https://raw.githubusercontent.com/COGNIMANEU/pilot03-service-job-card-extractor/main/install.ps1 | iex

# Dependencies: Windows always installs the CPU torch variant from the
# hash-pinned requirements\cpu.lock (PyPI has no CUDA 13 torch wheel for
# Windows, and install.sh probes nvidia-smi only on Linux). Set
# $env:LOCK_FILE = "<path>" (relative to the checkout) to choose another lock.

$ErrorActionPreference = "Stop"

$REPO_URL = "https://github.com/COGNIMANEU/pilot03-service-job-card-extractor.git"
$CPU_LOCK = "requirements/cpu.lock"
$CUDA_LOCK = "requirements.lock"

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

# Returns the dependency file to install, relative to the checkout.
function Select-LockFile {
    param($Checkout)
    $reason = "(Windows has no CUDA 13 torch build)"
    if ($env:LOCK_FILE) {
        if (-not (Test-Path (Join-Path $Checkout $env:LOCK_FILE))) {
            Write-Err "LOCK_FILE=$($env:LOCK_FILE) not found in $Checkout"
        }
        Write-Info "torch variant: LOCK_FILE=$($env:LOCK_FILE) (auto would pick cpu $reason)"
        if (($env:LOCK_FILE -replace '\\', '/' -replace '^\./', '') -eq $CUDA_LOCK) {
            Write-Warn "Installing the CUDA lock although auto picks cpu $reason; torch may not use this GPU"
        }
        return $env:LOCK_FILE
    }
    Write-Info "torch variant: cpu $reason"
    if (Test-Path (Join-Path $Checkout $CPU_LOCK)) { return $CPU_LOCK }
    if (Test-Path (Join-Path $Checkout $CUDA_LOCK)) {
        Write-Warn "$CPU_LOCK not found; installing $CUDA_LOCK instead"
        return $CUDA_LOCK
    }
    Write-Warn "No lock file found; installing unpinned requirements.txt"
    return "requirements.txt"
}

Write-Info "Installing Job Card Extractor..."

# Check Python
$pythonVersion = Get-PythonVersion
if (-not $pythonVersion) {
    Write-Err "Python 3.13+ not found. Install from https://www.python.org/downloads/"
}
Write-Info "Python $pythonVersion found"
if ([version]$pythonVersion -lt [version]"3.13") {
    Write-Err "Python 3.13+ required (found: $pythonVersion)"
}
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

$lockName = Select-LockFile $checkout
$lockFile = Join-Path $checkout $lockName
Write-Info "Installing Python packages from $lockName..."
if ($lockName -like "*.lock") {
    & $venvPython -m pip install --require-hashes -r $lockFile | Out-Null
}
else {
    & $venvPython -m pip install -r $lockFile | Out-Null
}
if ($LASTEXITCODE -ne 0) { Write-Err "Failed to install Python dependencies" }

Write-Ok "Installation complete!"
Write-Host ""
Write-Host "To activate the virtual environment, run:"
Write-Host "  $venvPath\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then run:"
Write-Host "  python $checkout\job_card_extractor.py <input.pdf> -o <output_dir>"