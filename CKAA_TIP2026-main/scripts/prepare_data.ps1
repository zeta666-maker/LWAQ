# Build the processed fault_csv dataset from the raw CSV streams.
#   Usage:  .\scripts\prepare_data.ps1
#   Env:    DATA_RAW_ROOT  directory containing data_1_S1/ and data_2_S1/
#                          (default: the parent directory of this repository)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")   # repository root

$RawRoot    = if ($env:DATA_RAW_ROOT) { $env:DATA_RAW_ROOT } else { ".." }
$OutputRoot = if ($env:OUTPUT_ROOT)   { $env:OUTPUT_ROOT }   else { "A_CLData/fault_csv" }

Write-Host "Building fault_csv from: $RawRoot"
python tools/prepare_tabular_dataset.py --raw-root $RawRoot --output-root $OutputRoot --overwrite
