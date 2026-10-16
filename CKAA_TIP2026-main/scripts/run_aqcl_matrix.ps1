# AQCL (paper 2): TC-MoA comparison matrix (fp32 / fixed8 / fixed4 / fixed2).
#   Usage:  .\scripts\run_aqcl_matrix.ps1
#   Env:    AQCL_CASES   comma-separated cases   (default: fp32,fixed8,fixed4,fixed2)
#           AQCL_SEEDS   comma-separated seeds   (default: 2024,2025)
#           AQCL_EPOCHS  epochs per task         (default: 8)
#           AQCL_PREFIX  log filename prefix     (default: tc)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")   # repository root

$Cases  = if ($env:AQCL_CASES)  { $env:AQCL_CASES }  else { "fp32,fixed8,fixed4,fixed2" }
$Seeds  = if ($env:AQCL_SEEDS)  { $env:AQCL_SEEDS }  else { "2024,2025" }
$Epochs = if ($env:AQCL_EPOCHS) { $env:AQCL_EPOCHS } else { "8" }
$Prefix = if ($env:AQCL_PREFIX) { $env:AQCL_PREFIX } else { "tc" }

python tools/run_aqcl_tcmoa_matrix.py --cases $Cases --seeds $Seeds --epochs $Epochs --prefix $Prefix
