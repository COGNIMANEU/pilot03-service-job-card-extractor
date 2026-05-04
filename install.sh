#!/usr/bin/env bash
set -euo pipefail

TOOL_NAME="job-card-extractor"
TOOL_REPO="COGNIMANEU/pilot03-service-job-card-extractor"
PYTHON_MIN_VERSION="3.6"

info() { echo "[INFO]  $*"; }
ok() { echo "[ OK ]  $*"; }
warn() { echo "[WARN]  $*"; }
err() { echo "[ERR]   $*"; }
die() { err "$*"; exit 1; }

detect_os() {
    case "${OSTYPE}" in
        darwin*) echo "macos" ;;
        linux*) echo "linux" ;;
        msys*|cygwin*) echo "windows" ;;
        *) echo "unknown" ;;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        x86_64) echo "x86_64" ;;
        aarch64|arm64) echo "arm64" ;;
        *) echo "unknown" ;;
    esac
}

detect_package_manager() {
    if command -v brew &>/dev/null; then
        echo "brew"
    elif command -v apt-get &>/dev/null; then
        echo "apt"
    elif command -v dnf &>/dev/null; then
        echo "dnf"
    elif command -v yum &>/dev/null; then
        echo "yum"
    elif command -v pacman &>/dev/null; then
        echo "pacman"
    elif command -v zypper &>/dev/null; then
        echo "zypper"
    else
        echo "none"
    fi
}

need_sudo() {
    if [[ $EUID -eq 0 ]]; then
        echo "false"
    elif command -v sudo &>/dev/null; then
        echo "true"
    else
        echo "false"
    fi
}

install_deps() {
    local os="$1"
    local pkg_mgr="$2"
    local with_sudo="$3"

    info "Installing dependencies..."

    local install_cmd=""
    local pkgs=""

    case "$os" in
        macos)
            if ! command -v brew &>/dev/null; then
                die "Homebrew not found. Install from https://brew.sh"
            fi
            pkgs="poppler"
            if [[ "$with_sudo" == "true" ]]; then
                install_cmd="sudo brew install"
            else
                install_cmd="brew install"
            fi
            ;;
        linux)
            pkgs="poppler-utils"
            case "$pkg_mgr" in
                apt)
                    if [[ "$with_sudo" == "true" ]]; then
                        install_cmd="sudo apt-get update && sudo apt-get install -y"
                    else
                        install_cmd="apt-get update && apt-get install -y"
                    fi
                    ;;
                dnf|yum)
                    if [[ "$with_sudo" == "true" ]]; then
                        install_cmd="sudo $pkg_mgr install -y"
                    else
                        install_cmd="$pkg_mgr install -y"
                    fi
                    ;;
                pacman)
                    if [[ "$with_sudo" == "true" ]]; then
                        install_cmd="sudo pacman -S --noconfirm"
                    else
                        install_cmd="pacman -S --noconfirm"
                    fi
                    ;;
                zypper)
                    if [[ "$with_sudo" == "true" ]]; then
                        install_cmd="sudo zypper install -y"
                    else
                        install_cmd="zypper install -y"
                    fi
                    ;;
                *)
                    die "No supported package manager found"
                    ;;
            esac
            ;;
        windows)
            warn "Windows detected. Use install.ps1 for PowerShell installation."
            die "Use install.ps1 on Windows"
            ;;
        *)
            die "Unsupported OS: $os"
            ;;
    esac

    if [[ -n "$install_cmd" ]]; then
        info "Running: $install_cmd $pkgs"
        eval "$install_cmd $pkgs" || die "Failed to install system dependencies"
    fi

    ok "Dependencies installed"
}

check_python() {
    if command -v python3 &>/dev/null; then
        local version
        version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        ok "Python $version found"
    else
        die "Python 3 not found. Please install Python $PYTHON_MIN_VERSION+"
    fi
}

install_python_deps() {
    info "Installing Python dependencies..."

    local venv_dir="${HOME}/.venv/${TOOL_NAME}"
    local pip_cmd

    if [[ -d "$venv_dir" ]]; then
        info "Using existing virtual environment"
        pip_cmd="${venv_dir}/bin/pip"
    else
        info "Creating virtual environment at $venv_dir"
        python3 -m venv "$venv_dir" || die "Failed to create virtual environment"
        pip_cmd="${venv_dir}/bin/pip"
    fi

    "$pip_cmd" install --upgrade pip >/dev/null 2>&1 || die "Failed to upgrade pip"

    local deps=(
        "numpy>=1.19.0"
        "opencv-python>=4.5.0"
        "Pillow>=8.0.0"
        "pdf2image>=1.16.0"
        "pyzbar>=0.1.8"
        "easyocr>=1.4.1"
        "torch>=1.7.0"
        "torchvision>=0.8.0"
    )

    info "Installing ${#deps[@]} packages..."
    "$pip_cmd" install "${deps[@]}" || die "Failed to install Python dependencies"

    ok "Python dependencies installed"

    if [[ -d "$venv_dir" ]]; then
        echo ""
        echo "============================================"
        echo "To activate the virtual environment, run:"
        echo "  source $venv_dir/bin/activate"
        echo "============================================"
    fi
}

verify_installation() {
    info "Verifying installation..."

    if command -v pdfinfo &>/dev/null; then
        ok "poppler installed"
    else
        die "poppler not found in PATH"
    fi

    if python3 -c "import cv2; import easyocr; import pyzbar; from pdf2image import convert_from_path" 2>/dev/null; then
        ok "Python packages installed"
    else
        die "Python packages not properly installed"
    fi

    ok "Verification complete"
}

main() {
    local os
    local arch
    local pkg_mgr
    local with_sudo

    os=$(detect_os)
    arch=$(detect_arch)
    pkg_mgr=$(detect_package_manager)
    with_sudo=$(need_sudo)

    info "OS: $os | Arch: $arch | Package Manager: $pkg_mgr"

    check_python
    install_deps "$os" "$pkg_mgr" "$with_sudo"
    install_python_deps
    verify_installation

    echo ""
    echo "============================================"
    ok "Installation complete!"
    echo ""
    echo "Usage:"
    echo "  source ~/.venv/${TOOL_NAME}/bin/activate"
    echo "  python job_card_extractor.py <input.pdf> -o <output_dir>"
    echo "============================================"
}

main "$@"