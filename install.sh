#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Job Card Extractor Installer
# Extracts job numbers and operations from manufacturing job card PDFs
# Usage: curl -sSL https://raw.githubusercontent.com/COGNIMANEU/pilot03-service-job-card-extractor/main/install.sh | bash
# Dependencies: installs from a hash-pinned lock chosen by the torch variant:
#   requirements/cpu.lock  CPU-only torch: macOS, and Linux without a GPU that
#                          can run the CUDA 13 build (the default on Pilot 03)
#   requirements.lock      CUDA 13 torch on Linux (driver >= 580, compute
#                          capability >= 7.5)
#   LOCK_FILE=<path>       overrides the choice; path relative to the checkout,
#                          e.g. LOCK_FILE=requirements.lock bash install.sh
# ============================================================================

# --- Configuration ---
REPO_URL="https://github.com/COGNIMANEU/pilot03-service-job-card-extractor.git"
PYTHON_MIN_VERSION="3.13"

# --- Torch variant minimums (the one place to change them) ---
# requirements.lock pins torch built for CUDA 13 (cu13x wheels) on Linux.
# - Driver: CUDA 13.x runs on NVIDIA driver >= 580 (CUDA Toolkit release notes,
#   "CUDA Toolkit and Corresponding Driver Versions": CUDA 13.0 ships with R580
#   and 13.x minor-version compatibility requires >= 580),
#   https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
# - Compute capability: CUDA 13 dropped Maxwell, Pascal and Volta, and the
#   PyTorch CUDA 13 wheels are built from Turing (7.5) up (RELEASE.md,
#   "PyTorch CUDA Support Matrix": "Turing(7.5), Ampere(8.0, 8.6), ..."),
#   https://github.com/pytorch/pytorch/blob/main/RELEASE.md
TORCH_CUDA_MAJOR=13
MIN_DRIVER_MAJOR=580
MIN_COMPUTE_CAP_MAJOR=7
MIN_COMPUTE_CAP_MINOR=5
CPU_LOCK="requirements/cpu.lock"
CUDA_LOCK="requirements.lock"

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

# --- Torch variant ---
# Prints "<cpu|cuda> (<reason>)" on stdout and never fails. "cuda" only on
# Linux when nvidia-smi succeeds and some GPU meets both minimums above;
# otherwise the reason names the first GPU's shortfall.
torch_variant() {
    local os="$1" out driver cc dmaj cmaj cmin reason=""
    if [ "$os" != "linux" ]; then
        echo "cpu ($os has no CUDA $TORCH_CUDA_MAJOR torch build)"; return 0
    fi
    if ! command -v nvidia-smi &>/dev/null; then
        echo "cpu (nvidia-smi not found)"; return 0
    fi
    # A wedged driver can hang nvidia-smi; bound it where timeout(1) exists.
    local runner=()
    command -v timeout &>/dev/null && runner=(timeout 20)
    if ! out="$("${runner[@]+"${runner[@]}"}" nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader 2>/dev/null)" ||
        [ -z "$out" ]; then
        echo "cpu (nvidia-smi failed)"; return 0
    fi
    while IFS=, read -r driver cc; do
        driver="${driver//[[:space:]]/}"; cc="${cc//[[:space:]]/}"
        dmaj="${driver%%.*}"; cmaj="${cc%%.*}"; cmin="${cc#*.}"
        if ! [[ "$cc" == *.* && "$dmaj" =~ ^[0-9]+$ && "$cmaj" =~ ^[0-9]+$ && "$cmin" =~ ^[0-9]+$ ]]; then
            reason="${reason:-unreadable nvidia-smi output: $driver, $cc}"; continue
        fi
        if (( 10#$dmaj < MIN_DRIVER_MAJOR )); then
            reason="${reason:-driver $dmaj < $MIN_DRIVER_MAJOR required by CUDA $TORCH_CUDA_MAJOR}"; continue
        fi
        if (( 10#$cmaj < MIN_COMPUTE_CAP_MAJOR ||
              (10#$cmaj == MIN_COMPUTE_CAP_MAJOR && 10#$cmin < MIN_COMPUTE_CAP_MINOR) )); then
            reason="${reason:-compute capability $cc < $MIN_COMPUTE_CAP_MAJOR.$MIN_COMPUTE_CAP_MINOR required by CUDA $TORCH_CUDA_MAJOR torch}"; continue
        fi
        echo "cuda (driver $dmaj, compute capability $cc)"; return 0
    done <<< "$out"
    echo "cpu (${reason:-no GPU reported by nvidia-smi})"
}

# Prints the dependency file to install, relative to the checkout, on stdout;
# messages go to stderr so callers can capture the path.
select_lockfile() {
    local checkout="$1" os="$2" decision variant reason
    decision="$(torch_variant "$os")"
    variant="${decision%% *}"; reason="${decision#* }"

    if [ -n "${LOCK_FILE:-}" ]; then
        [ -f "$checkout/$LOCK_FILE" ] || die "LOCK_FILE=$LOCK_FILE not found in $checkout"
        info "torch variant: LOCK_FILE=$LOCK_FILE (auto would pick $variant $reason)" >&2
        if [ "${LOCK_FILE#./}" = "$CUDA_LOCK" ] && [ "$variant" = "cpu" ]; then
            warn "Installing the CUDA lock although auto picks cpu $reason; torch may not use this GPU" >&2
        fi
        echo "$LOCK_FILE"; return 0
    fi

    info "torch variant: $variant $reason" >&2
    local preferred="$CPU_LOCK" other="$CUDA_LOCK"
    [ "$variant" = "cuda" ] && preferred="$CUDA_LOCK" other="$CPU_LOCK"
    if [ -f "$checkout/$preferred" ]; then
        echo "$preferred"
    elif [ -f "$checkout/$other" ]; then
        warn "$preferred not found; installing $other instead" >&2
        echo "$other"
    else
        warn "No lock file found; installing unpinned requirements.txt" >&2
        echo "requirements.txt"
    fi
}

# --- Python Environment Setup ---
install_job_card_extractor() {
    local checkout="$1" venv_dir="$1/venv" os="${2:-}" lockfile
    [ -n "$os" ] || os="$(detect_os)"
    # Choose (and validate LOCK_FILE) before the slow venv and pip steps.
    lockfile="$(select_lockfile "$checkout" "$os")"
    info "Setting up Python virtual environment at $venv_dir..."

    if [[ ! -d "$venv_dir" ]]; then
        python3 -m venv "$venv_dir" || die "Failed to create virtual environment"
    fi
    [[ -x "$venv_dir/bin/python" ]] || die "Virtual environment missing Python at $venv_dir/bin/python"

    info "Upgrading pip..."
    "$venv_dir/bin/python" -m pip install --upgrade pip >/dev/null 2>&1 || die "Failed to upgrade pip"

    # Hash-pinned locks target Python 3.13; requirements.txt is the loose fallback.
    if [[ "$lockfile" == *.lock ]]; then
        info "Installing Python packages from $lockfile (hash-verified)..."
        "$venv_dir/bin/python" -m pip install --require-hashes -r "$checkout/$lockfile" || die "Failed to install Python dependencies"
    else
        info "Installing Python packages..."
        "$venv_dir/bin/python" -m pip install -r "$checkout/$lockfile" || die "Failed to install Python dependencies"
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
    install_job_card_extractor "$checkout" "$os"
    verify_installation "$checkout"

    echo ""
    ok "Installation complete!"
}

if [[ "${BASH_SOURCE[0]-}" == "$0" || ! -f "${BASH_SOURCE[0]-}" ]]; then
    main "$@"
fi
