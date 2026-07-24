param(
    [string]$Seeds = "42",
    [string]$Device = "cuda",
    [int]$PackageSize = 2,
    [string]$OutputRoot = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = "$ScriptDir\runs\main_score_5k"
}

& $Python "$ScriptDir\run_main_text_experiments.py" `
    --preset main_score_5k `
    --seeds $Seeds `
    --device $Device `
    --package-size $PackageSize `
    --output-root $OutputRoot `
    --score-only
