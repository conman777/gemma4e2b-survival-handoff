param(
    [string]$ModelName = "google/gemma-4-E2B-it",
    [string]$DatasetPath = ".\data\normalized\survival_training_candidates_merged.jsonl",
    [string]$OutputDir = ".\output\gemma4e2b-survival-lora-curated",
    [int]$MaxSeqLength = 2048,
    [int]$Epochs = 1,
    [double]$LearningRate = 2e-4,
    [int]$BatchSize = 1,
    [int]$GradientAccumulation = 8,
    [int]$LoggingSteps = 10,
    [int]$SaveSteps = 500,
    [int]$EvalSteps = 200,
    [ValidateSet("no", "steps", "epoch")]
    [string]$EvaluationStrategy = "no",
    [ValidateSet("steps", "epoch")]
    [string]$SaveStrategy = "steps",
    [int]$LoraRank = 16
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment not found. Run .\scripts\setup_env.ps1 first."
}

& $pythonExe "$PSScriptRoot\train_gemma_survival.py" `
    --model-name $ModelName `
    --dataset-path (Join-Path $root $DatasetPath) `
    --output-dir (Join-Path $root $OutputDir) `
    --max-seq-length $MaxSeqLength `
    --num-train-epochs $Epochs `
    --per-device-train-batch-size $BatchSize `
    --gradient-accumulation-steps $GradientAccumulation `
    --logging-steps $LoggingSteps `
    --save-steps $SaveSteps `
    --eval-steps $EvalSteps `
    --evaluation-strategy $EvaluationStrategy `
    --save-strategy $SaveStrategy `
    --learning-rate $LearningRate `
    --lora-r $LoraRank
