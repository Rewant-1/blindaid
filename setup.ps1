param(
    [switch]$Dev
)

$cyan = "`e[36m"
$yellow = "`e[33m"
$green = "`e[32m"
$red = "`e[31m"
$reset = "`e[0m"

function Write-Info($message) { Write-Host "$cyan$message$reset" }
function Write-Warn($message) { Write-Host "$yellow$message$reset" }
function Write-Ok($message) { Write-Host "$green$message$reset" }
function Write-Err($message) { Write-Host "$red$message$reset" }

Write-Info "BlindAid setup"
Write-Host "====================`n"

$python = $Env:PYTHON
if (-not $python) { $python = "python" }

if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
    Write-Err "Python 3.9+ not found. Install Python and rerun this script."
    exit 1
}

if (-not (Test-Path ".venv")) {
    Write-Warn "Creating .venv"
    & $python -m venv .venv
} else {
    Write-Ok "Virtual environment already present (.venv)"
}

$venvActivate = Join-Path -Path ".venv" -ChildPath "Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Err "Could not locate $venvActivate"
    exit 1
}

Write-Warn "Activating virtual environment"
& $venvActivate

Write-Warn "Upgrading pip"
python -m pip install --upgrade pip wheel

Write-Warn "Installing runtime dependencies"
pip install -r requirements.txt

if ($Dev.IsPresent) {
    Write-Warn "Installing developer dependencies"
    pip install -r requirements-dev.txt
}

Write-Ok "Dependencies installed."
Write-Info "Copy your YOLO weights into resources/models and face images into resources/known_faces."
Write-Info "Launch with: python -m blindaid"
