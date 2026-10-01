#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Job Card Extractor Installer
# Extracts job numbers and operations from manufacturing job card PDFs
# Usage: curl -sSL https://raw.githubusercontent.com/COGNIMANEU/pilot03-service-job-card-extractor/main/install.sh | bash
# ============================================================================

# --- Configuration ---
REPO_URL="https://github.com/COGNIMANEU/pilot03-service-job-card-extractor.git"
PYTHON_MIN_VERSION="3.13"

# --- Color Output ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

info()  { printf "${BLUE}[INFO]${NC}  %s\n" "$*"; }
ok()    { printf "${GREEN}[ OK ]${NC}  %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()   { printf "${RED}[ERR ]${NC}  %s\n" "$*" >&2; }
die()   { err "$@"; exit 1; }

# --- OS Detection ---
detect_os() {
    local os
    os="$(uname -s | tr '[:upper:]' '[:lower:]')"
    case "$os" in
        linux*)  echo "linux" ;;
        darwin*) echo "macos" ;;
        mingw*|msys*|cygwin*) echo "windows" ;;
        *)       die "Unsupported operating system: $os" ;;
    esac
}

# --- Architecture Detection ---
detect_arch() {
    local arch
    arch="$(uname -m)"
    case "$arch" in
        x86_64|amd64)  echo "x86_64" ;;
        aarch64|arm64) echo "arm64" ;;
        *)             die "Unsupported architecture: $arch" ;;
    esac
}

# --- Package Manager Detection ---
# On macOS, prefer Homebrew. On Linux, prefer the native system package manager
# over Homebrew (Linuxbrew) so system dependencies install where the OS expects.
detect_package_manager() {
    local os="$1"
    if [ "$os" = "macos" ]; then
        if command -v brew &>/dev/null; then echo "brew"; else echo "unknown"; fi
        return
    fi
    if   command -v apt-get &>/dev/null; then echo "apt"
    elif command -v dnf     &>/dev/null; then echo "dnf"
    elif command -v yum     &>/dev/null; then echo "yum"
    elif command -v pacman  &>/dev/null; then echo "pacman"
    elif command -v zypper  &>/dev/null; then echo "zypper"
    elif command -v brew    &>/dev/null; then echo "brew"
    else echo "unknown"
    fi
}

# --- System Dependencies ---
# None: pypdfium2 (PDF rendering) and zxing-cpp (barcode decoding) ship
# self-contained wheels, so no system packages are needed. See
# docs/decisions/ocr-stack-spike.md.

# --- Python Version Check ---
check_python() {
    if command -v python3 &>/dev/null; then
        local version
        version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        info "Python $version found"
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,13) else 1)" 2>/dev/null; then
            ok "Python $version meets minimum requirement ($PYTHON_MIN_VERSION+)"
        else
            die "Python $PYTHON_MIN_VERSION+ required (found: $version)"
        fi
    else
        die "Python 3 not found. Install from https://www.python.org/downloads/"
    fi
}

# A downloaded script has no checkout path; clone into the caller's directory.
# A local invocation instead uses the checkout containing the script.
resolve_checkout() {
    local script_dir="" checkout
    if [[ -f "${BASH_SOURCE[0]-}" ]]; then
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    fi
    if [[ -n "$script_dir" && -f "$script_dir/job_card_extractor.py" ]]; then
        checkout="$script_dir"
    elif [[ -f "$PWD/job_card_extractor.py" ]]; then
        checkout="$PWD"
    else
        checkout="$PWD/pilot03-service-job-card-extractor"
        if [[ ! -e "$checkout" ]]; then
            command -v git &>/dev/null || die "Git is required to download the extractor"
            git clone "$REPO_URL" "$checkout" || die "Failed to clone the extractor"
        fi
    fi
    [[ -f "$checkout/job_card_extractor.py" && -f "$checkout/requirements.txt" ]] ||
        die "Extractor checkout incomplete at $checkout"
    printf '%s\n' "$checkout"
}

# --- Python Environment Setup ---
install_job_card_extractor() {
    local checkout="$1" venv_dir="$1/venv"
    info "Setting up Python virtual environment at $venv_dir..."

    if [[ ! -d "$venv_dir" ]]; then
        python3 -m venv "$venv_dir" || die "Failed to create virtual environment"
    fi
    [[ -x "$venv_dir/bin/python" ]] || die "Virtual environment missing Python at $venv_dir/bin/python"

    info "Upgrading pip..."
    "$venv_dir/bin/python" -m pip install --upgrade pip >/dev/null 2>&1 || die "Failed to upgrade pip"

    # Prefer the hash-pinned lock (Python 3.13); fall back to the loose floors.
    if [[ -f "$checkout/requirements.lock" ]]; then
        info "Installing Python packages from requirements.lock (hash-verified)..."
        "$venv_dir/bin/python" -m pip install --require-hashes -r "$checkout/requirements.lock" || die "Failed to install Python dependencies"
    else
        info "Installing Python packages..."
        "$venv_dir/bin/python" -m pip install -r "$checkout/requirements.txt" || die "Failed to install Python dependencies"
    fi

    ok "Python dependencies installed"

    cat << ACTIVATE_HELP

============================================
To activate the virtual environment, run:

  source ${venv_dir}/bin/activate

Then use the tool:

  python ${checkout}/job_card_extractor.py <input.pdf> -o <output_dir>
============================================
ACTIVATE_HELP
}

# --- Verification ---
verify_installation() {
    local checkout="$1"
    info "Verifying installation..."

    if "$checkout/venv/bin/python" -c "import cv2, easyocr, zxingcpp, pypdfium2" 2>/dev/null; then
        ok "Python packages importable"
    else
        die "Python packages not properly installed. Activate the venv and check with: pip list"
    fi

    ok "Installation verified successfully"
}

# --- Main ---
main() {
    local os arch pm checkout

    os=$(detect_os)
    arch=$(detect_arch)

    if [ "$os" = "windows" ]; then
        die "Windows is not supported by this script. Run the PowerShell installer instead:
  irm https://raw.githubusercontent.com/COGNIMANEU/pilot03-service-job-card-extractor/main/install.ps1 | iex"
    fi

    pm=$(detect_package_manager "$os")

    info "OS: $os | Arch: $arch | Package Manager: $pm"

    check_python
    checkout=$(resolve_checkout)
    install_job_card_extractor "$checkout"
    verify_installation "$checkout"

    echo ""
    ok "Installation complete!"
}

if [[ "${BASH_SOURCE[0]-}" == "$0" || ! -f "${BASH_SOURCE[0]-}" ]]; then
    main "$@"
fi
