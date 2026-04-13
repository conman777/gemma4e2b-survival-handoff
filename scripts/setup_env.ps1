param(
    [string]$VenvPath = ".venv",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venvFullPath = Join-Path $root $VenvPath

Write-Host "Creating virtual environment at $venvFullPath"
python -m venv $venvFullPath

$pythonExe = Join-Path $venvFullPath "Scripts\python.exe"
$pipArgsBase = @("-m", "pip")

Write-Host "Upgrading pip tooling"
& $pythonExe @pipArgsBase install --upgrade pip setuptools wheel

Write-Host "Installing PyTorch from $TorchIndexUrl"
& $pythonExe @pipArgsBase install torch torchvision torchaudio --index-url $TorchIndexUrl

Write-Host "Installing Unsloth and training dependencies"
& $pythonExe @pipArgsBase install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"
& $pythonExe @pipArgsBase install datasets trl accelerate peft bitsandbytes sentencepiece protobuf huggingface_hub

Write-Host "Reinstalling CUDA-enabled PyTorch after dependency resolution"
& $pythonExe @pipArgsBase install --force-reinstall torch torchvision torchaudio --index-url $TorchIndexUrl

Write-Host ""
Write-Host "Environment ready."
Write-Host "Next:"
Write-Host "  1. .\scripts\login_hf.ps1"
Write-Host "  2. .\scripts\run_train.ps1"
