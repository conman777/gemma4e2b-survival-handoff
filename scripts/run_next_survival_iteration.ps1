param(
    [string]$ModelName = "google/gemma-4-E2B-it",
    [string]$DatasetPath = ".\data\normalized\survival_training_candidates_merged.jsonl",
    [string]$AdapterDir = ".\output\gemma4e2b-survival-lora-next",
    [string]$EvalOutputDir = ".\output\evals",
    [int]$Epochs = 1,
    [double]$LearningRate = 1e-4,
    [int]$LoraRank = 16,
    [switch]$SkipNormalize,
    [switch]$SkipTrain
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment not found. Run .\scripts\setup_env.ps1 first."
}

if (-not $SkipNormalize) {
    & $pythonExe "$PSScriptRoot\normalize_open_survival_sources.py"
}

if (-not $SkipTrain) {
    & "$PSScriptRoot\run_train.ps1" `
        -ModelName $ModelName `
        -DatasetPath $DatasetPath `
        -OutputDir $AdapterDir `
        -Epochs $Epochs `
        -LearningRate $LearningRate `
        -LoraRank $LoraRank
}

$defaultEval = Join-Path $root "data\evals\survival_eval_suite.jsonl"
$holdoutEval = Join-Path $root "data\evals\survival_eval_holdout_v1.jsonl"
$resolvedAdapterDir = Join-Path $root $AdapterDir
$resolvedEvalOutputDir = Join-Path $root $EvalOutputDir

& $pythonExe "$PSScriptRoot\run_survival_eval.py" `
    --adapter-dir $resolvedAdapterDir `
    --eval-file $defaultEval `
    --output-dir $resolvedEvalOutputDir

& $pythonExe "$PSScriptRoot\run_survival_eval.py" `
    --adapter-dir $resolvedAdapterDir `
    --eval-file $holdoutEval `
    --output-dir $resolvedEvalOutputDir
