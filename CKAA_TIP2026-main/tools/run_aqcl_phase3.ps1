# Phase-3 focused ablation: why does 4-bit AQCL lose accuracy on fault_csv?
#
# Each case runs 2 tasks (task 1 + task 2). Task 0 is identical across cases, so
# task-1 accuracy isolates the effect of the quantization scope/floor and task-2
# accuracy shows whether the effect compounds through continual learning.
#
# Usage:
#   .\tools\run_aqcl_phase3.ps1
#   .\tools\run_aqcl_phase3.ps1 -Cases full_4_8,scope_backbone_fp32
param(
    [string[]] $Cases = @(
        "full_4_8",
        "scope_backbone_fp32",
        "floor8_4_8",
        "lr03_full_4_8"
    ),
    [int] $Epochs = 2,
    [int] $WarmupEpochs = 1,
    [int] $FisherBatches = 2,
    [int] $MaxTasks = 2,
    [string] $SuffixPrefix = "p3"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$OutDir = Split-Path $Root -Parent
$WorkDir = Join-Path $OutDir "_aqcl_work"
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

# case -> env overrides and extra CLI flags
$CaseTable = @{
    full_4_8            = @{ Env = @{};                                       Flags = @() }
    scope_backbone_fp32 = @{ Env = @{ AQCL_QUANTIZE_BACKBONE = "0" };         Flags = @() }
    floor8_4_8          = @{ Env = @{};                                       Flags = @("--aqcl-backbone-bits", "8") }
    floor16_4_8         = @{ Env = @{};                                       Flags = @("--aqcl-backbone-bits", "16") }
    lr03_full_4_8       = @{ Env = @{};                                       Flags = @("--lr", "0.0015") }
    e4_full_4_8         = @{ Env = @{};                                       Flags = @() }
}

foreach ($case in $Cases) {
    if (-not $CaseTable.ContainsKey($case)) {
        throw "Unknown case '$case'. Known: $($CaseTable.Keys -join ', ')"
    }
    $spec = $CaseTable[$case]
    $caseEpochs = $Epochs
    if ($case -like "e4_*") { $caseEpochs = 4 }
    $suffix = "${SuffixPrefix}_$case"
    Write-Host "=== Phase-3 case: $case ===" -ForegroundColor Cyan

    foreach ($key in $spec.Env.Keys) {
        Set-Item -Path "Env:$key" -Value $spec.Env[$key]
    }

    $arguments = @(
        (Join-Path $Root "train_aqcl.py"),
        "-d", "fault_csv",
        "-t", "5",
        "-m", "vit_base_patch16_224.augreg_in21k",
        "-b", "32",
        "-e", "$caseEpochs",
        "-jt", "0", "-je", "0", "-et", "1",
        "--eval_batch_size", "64",
        "--temperature", "28.0",
        "--lr", "0.005",
        "--lr_scale", "0.2",
        "--lr_scale_patterns", "patch_embed",
        "--null_eta1", "0.999",
        "--null_eta2", "0.999",
        "-tc", "2.0",
        "--eval-tool", "adapter",
        "--eval-trained-task-router", "true",
        "--eval-local-task-head", "true",
        "--eval-task-weight", "false",
        "--prototype-confidence-weight", "0.0",
        "--seed", "2024",
        "--save-model", "false",
        "--logs-dir", "logs",
        "-sf", $suffix,
        "--aqcl-mode", "rpq_saou",
        "--aqcl-low-bits", "4",
        "--aqcl-high-bits", "8",
        "--aqcl-fisher-batches", "$FisherBatches",
        "--aqcl-warmup-epochs", "$WarmupEpochs",
        "--aqcl-instrument", "true",
        "--aqcl-instrument-verbose", "false",
        "--aqcl-debug-max-tasks", "$MaxTasks"
    ) + $spec.Flags

    $stdout = Join-Path $WorkDir "$suffix.out"
    $stderr = Join-Path $WorkDir "$suffix.err"
    $process = Start-Process -NoNewWindow -PassThru -Wait -FilePath "python" `
        -ArgumentList (@("-u") + $arguments) `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Write-Host "  exit=$($process.ExitCode) log=$Root\logs\$suffix.txt"

    foreach ($key in $spec.Env.Keys) {
        Remove-Item -Path "Env:$key" -ErrorAction SilentlyContinue
    }
    if ($process.ExitCode -ne 0) {
        Get-Content $stderr -Tail 20
        throw "Case '$case' failed with exit code $($process.ExitCode)"
    }
}
