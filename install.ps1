# Job Card Extractor - Windows Installation Script
# Run with: irm https://raw.githubusercontent.com/COGNIMANEU/pilot03-service-job-card-extractor/main/install.ps1 | iex

$ErrorActionPreference = "Stop"

$TOOL_NAME = "job-card-extractor"
$REPO = "COGNIMANEU/pilot03-service-job-card-extractor"

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

# Create virtual environment
$venvPath = "$env:USERPROFILE\.venv\$TOOL_NAME"
if (Test-Path $venvPath) {
    Write-Info "Using existing virtual environment"
}
else {
    Write-Info "Creating virtual environment at $venvPath"
    python -m venv $venvPath
}
$pip = "$venvPath\Scripts\pip"

# Upgrade pip
Write-Info "Upgrading pip..."
& $pip install --upgrade pip | Out-Null

# Install dependencies
$deps = @(
    "numpy>=1.19.0",
    "opencv-python>=4.5.0",
    "Pillow>=8.0.0",
    "pdf2image>=1.16.0",
    "pyzbar>=0.1.8",
    "easyocr>=1.4.1",
    "torch>=1.7.0",
    "torchvision>=0.8.0"
)

Write-Info "Installing Python packages..."
& $pip install $deps | Out-Null

Write-Ok "Installation complete!"
Write-Host ""
Write-Host "To activate the virtual environment, run:"
Write-Host "  $venvPath\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then run:"
Write-Host "  python job_card_extractor.py <input.pdf> -o <output_dir>"