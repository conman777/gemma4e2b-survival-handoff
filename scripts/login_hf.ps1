$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Python environment not found. Run .\scripts\setup_env.ps1 first."
}

& $pythonExe -c "from huggingface_hub import login; login()"
