param(
    [string]$Seeds = "42",
    [string]$Ratios = "0.0025,0.005,0.01,0.02",
    [string]$Device = "cuda",
    [int]$PackageSize = 2,
    [string]$OutputRoot = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = "$ScriptDir\runs\ratio_downstream_5k"
}

& $Python "$ScriptDir\run_ratio_downstream_training.py" `
    --preset ratio_5k `
    --seeds $Seeds `
    --ratios $Ratios `
    --device $Device `
    --package-size $PackageSize `
    --output-root $OutputRoot
