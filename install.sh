#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Job Card Extractor Installer
# Extracts job numbers and operations from manufacturing job card PDFs
# Usage: curl -sSL https://raw.githubusercontent.com/COGNIMANEU/pilot03-service-job-card-extractor/main/install.sh | bash
# ============================================================================

# --- Configuration ---
TOOL_NAME="job-card-extractor"
PYTHON_MIN_VERSION="3.6"

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
detect_package_manager() {
    if   command -v apt-get &>/dev/null; then echo "apt"
    elif command -v dnf     &>/dev/null; then echo "dnf"
    elif command -v yum     &>/dev/null; then echo "yum"
    elif command -v pacman  &>/dev/null; then echo "pacman"
    elif command -v brew    &>/dev/null; then echo "brew"
    elif command -v zypper  &>/dev/null; then echo "zypper"
    else echo "unknown"
    fi
}

# --- Sudo Detection ---
need_sudo() {
    if [ "$(id -u)" -ne 0 ]; then
        if command -v sudo &>/dev/null; then
            echo "sudo"
        else
            die "Root privileges required. Run as root or install sudo."
        fi
    else
        echo ""
    fi
}

# --- System Dependencies (Poppler) ---
install_deps() {
    local pm="$1" sudo_cmd="$2"
    info "Installing system dependencies..."

    case "$pm" in
        brew)
            brew install poppler || die "Failed to install poppler"
            ;;
        apt)
            $sudo_cmd apt-get update -qq && $sudo_cmd apt-get install -y -qq poppler-utils || die "Failed to install poppler-utils"
            ;;
        dnf|yum)
            $sudo_cmd "$pm" install -y -q poppler-utils || die "Failed to install poppler-utils"
            ;;
        pacman)
            $sudo_cmd pacman -Sy --noconfirm poppler || die "Failed to install poppler"
            ;;
        zypper)
            $sudo_cmd zypper install -y poppler-tools || die "Failed to install poppler-tools"
            ;;
        *)
            die "Unsupported package manager '$pm'. Install poppler manually: https://poppler.freedesktop.org/"
            ;;
    esac

    ok "System dependencies installed"
}

# --- Python Version Check ---
check_python() {
    if command -v python3 &>/dev/null; then
        local version
        version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        info "Python $version found"
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,6) else 1)" 2>/dev/null; then
            ok "Python $version meets minimum requirement ($PYTHON_MIN_VERSION+)"
        else
            die "Python 3.6+ required (found: $version)"
        fi
    else
        die "Python 3 not found. Install from https://www.python.org/downloads/"
    fi
}

# --- Python Environment Setup ---
install_job_card_extractor() {
    info "Setting up Python virtual environment..."

    local venv_dir="${HOME}/.venv/${TOOL_NAME}"
    local pip_cmd

    if [[ -d "$venv_dir" ]]; then
        info "Using existing virtual environment at $venv_dir"
        pip_cmd="${venv_dir}/bin/pip"
    else
        info "Creating virtual environment at $venv_dir"
        python3 -m venv "$venv_dir" || die "Failed to create virtual environment"
        pip_cmd="${venv_dir}/bin/pip"
    fi

    info "Upgrading pip..."
    "$pip_cmd" install --upgrade pip >/dev/null 2>&1 || die "Failed to upgrade pip"

    info "Installing Python packages..."
    "$pip_cmd" install \
        "numpy>=1.19.0" \
        "opencv-python>=4.5.0" \
        "Pillow>=8.0.0" \
        "pdf2image>=1.16.0" \
        "pyzbar>=0.1.8" \
        "easyocr>=1.4.1" \
        "torch>=1.7.0" \
        "torchvision>=0.8.0" \
        || die "Failed to install Python dependencies"

    ok "Python dependencies installed"

    cat << ACTIVATE_HELP

============================================
To activate the virtual environment, run:

  source ${venv_dir}/bin/activate

Then use the tool:

  python job_card_extractor.py <input.pdf> -o <output_dir>
============================================
ACTIVATE_HELP
}

# --- Verification ---
verify_installation() {
    info "Verifying installation..."

    if command -v pdfinfo &>/dev/null; then
        ok "Poppler installed"
    else
        die "Poppler not found in PATH. Check your package manager installation."
    fi

    if python3 -c "import cv2, easyocr, pyzbar, pdf2image" 2>/dev/null; then
        ok "Python packages importable"
    else
        die "Python packages not properly installed. Activate the venv and check with: pip list"
    fi

    ok "Installation verified successfully"
}

# --- Main ---
main() {
    local os arch pm sudo_cmd

    os=$(detect_os)
    arch=$(detect_arch)
    pm=$(detect_package_manager)
    sudo_cmd=$(need_sudo)

    info "OS: $os | Arch: $arch | Package Manager: $pm"

    check_python
    install_deps "$pm" "$sudo_cmd"
    install_job_card_extractor
    verify_installation

    echo ""
    ok "Installation complete!"
}

main "$@"
