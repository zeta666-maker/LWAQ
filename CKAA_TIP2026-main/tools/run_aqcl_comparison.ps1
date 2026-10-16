# Phase-4 AQCL comparison matrix on fault_csv (5 tasks, 2/2/2/2/1 split).
#
# Every case uses the identical CKAA configuration described in the handoff and
# only differs in the AQCL quantization settings:
#   * --aqcl-backbone-bits 8 keeps the frozen pretrained ViT backbone at 8-bit
#     (see AQCL.md: Eq.13 is applied to the CKAA trainable set because the
#     backbone is frozen here, unlike the paper's from-scratch ResNet-20).
#     Pass -BackboneBits 0 for the paper-faithful whole-network split.
#
# Usage (execution policy permitting):
#   .\tools\run_aqcl_comparison.ps1
#   .\tools\run_aqcl_comparison.ps1 -Cases fp32,fixed8,rpq_saou_4_8
#   .\tools\run_aqcl_comparison.ps1 -BackboneBits 0 -SuffixPrefix aqcl_paperfaithful
param(
    [string[]] $Cases = @(
        "fp32", "fixed8", "fixed4", "fixed2",
        "rpq_4_8", "rpq_saou_4_8", "rpq_saou_2_4"
    ),
    [int] $Epochs = 2,
    [int] $WarmupEpochs = 1,
    [int] $FisherBatches = 2,
    [int] $Tasks = 5,
    [int] $BackboneBits = 8,
    [switch] $Instrument,
    [switch] $PrepareData,
    [switch] $PaperFaithful,
    [string] $SuffixPrefix = "aqcl_compare"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path $PSScriptRoot -Parent
$RawRoot = Split-Path $Root -Parent
$DataRoot = Join-Path $Root "A_CLData\fault_csv"
$WorkDir = Join-Path $RawRoot "_aqcl_work"
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

if ($PaperFaithful) { $BackboneBits = 0 }

if ($PrepareData) {
    python (Join-Path $Root "tools\prepare_tabular_dataset.py") `
        --raw-root $RawRoot `
        --output-root $DataRoot
}

# Bits = fixed width, Low/High = RPQ b_L/b_H (b_H = 2 * b_L as in the paper).
$CaseTable = @{
    fp32          = @{ Aqcl = "false"; Mode = "fixed";    Bits = "32"; Low = "32"; High = "32" }
    fixed8        = @{ Aqcl = "true";  Mode = "fixed";    Bits = "8";  Low = "8";  High = "8" }
    fixed4        = @{ Aqcl = "true";  Mode = "fixed";    Bits = "4";  Low = "4";  High = "4" }
    fixed2        = @{ Aqcl = "true";  Mode = "fixed";    Bits = "2";  Low = "2";  High = "2" }
    rpq_8_16      = @{ Aqcl = "true";  Mode = "rpq";      Bits = "8";  Low = "8";  High = "16" }
    rpq_4_8       = @{ Aqcl = "true";  Mode = "rpq";      Bits = "4";  Low = "4";  High = "8" }
    rpq_saou_8_16 = @{ Aqcl = "true";  Mode = "rpq_saou"; Bits = "8";  Low = "8";  High = "16" }
    rpq_saou_4_8  = @{ Aqcl = "true";  Mode = "rpq_saou"; Bits = "4";  Low = "4";  High = "8" }
    rpq_saou_2_4  = @{ Aqcl = "true";  Mode = "rpq_saou"; Bits = "2";  Low = "2";  High = "4" }
}

foreach ($case in $Cases) {
    if (-not $CaseTable.ContainsKey($case)) {
        throw "Unknown case '$case'. Known: $($CaseTable.Keys -join ', ')"
    }
    $spec = $CaseTable[$case]
    $suffix = "${SuffixPrefix}_$case"
    Write-Host "=== AQCL case: $case (suffix $suffix) ===" -ForegroundColor Cyan

    $arguments = @(
        (Join-Path $Root "train_aqcl.py"),
        "-d", "fault_csv",
        "-t", "$Tasks",
        "-m", "vit_base_patch16_224.augreg_in21k",
        "-b", "32",
        "-e", "$Epochs",
        "-jt", "0",
        "-je", "0",
        "-et", "1",
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
        "--aqcl-enable", $spec.Aqcl,
        "--aqcl-mode", $spec.Mode,
        "--aqcl-bits", $spec.Bits,
        "--aqcl-low-bits", $spec.Low,
        "--aqcl-high-bits", $spec.High,
        "--aqcl-backbone-bits", "$BackboneBits",
        "--aqcl-fisher-batches", "$FisherBatches",
        "--aqcl-warmup-epochs", "$WarmupEpochs"
    )
    if ($Instrument) {
        $arguments += @("--aqcl-instrument", "true", "--aqcl-instrument-verbose", "false")
    }

    $stdout = Join-Path $WorkDir "$suffix.out"
    $stderr = Join-Path $WorkDir "$suffix.err"
    $process = Start-Process -NoNewWindow -PassThru -Wait -FilePath "python" `
        -ArgumentList (@("-u") + $arguments) `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Write-Host "  exit=$($process.ExitCode) log=$Root\logs\$suffix.txt"
    if ($process.ExitCode -ne 0) {
        Get-Content $stderr -Tail 25
        throw "Case '$case' failed with exit code $($process.ExitCode)"
    }
}
