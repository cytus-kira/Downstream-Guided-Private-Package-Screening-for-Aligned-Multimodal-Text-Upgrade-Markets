param(
    [string]$Seeds = "42,43,44",
    [string]$Ratios = "0.0025,0.005,0.01,0.02",
    [string]$Device = "cuda",
    [int]$PackageSize = 2,
    [string]$RunTag = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ScriptDir "runs\run_all_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
if ([string]::IsNullOrWhiteSpace($RunTag)) {
    $RunTag = "latest_online_krr_$Stamp"
}
$RunRoot = Join-Path $ScriptDir "runs\$RunTag"
$Main5kRoot = Join-Path $RunRoot "main_score_5k"
$AblationRoot = Join-Path $RunRoot "main_ablation_5k"
$RatioRoot = Join-Path $RunRoot "ratio_downstream_5k"
$Main20kRoot = Join-Path $RunRoot "main_score_20k"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$Transcript = Join-Path $LogDir "run_all_$Stamp.log"
Start-Transcript -Path $Transcript -Force | Out-Null

try {
    Write-Host "[run-all] started $(Get-Date -Format o)"
    Write-Host "[run-all] seeds=$Seeds ratios=$Ratios device=$Device package_size=$PackageSize run_tag=$RunTag"
    Write-Host "[run-all] run_root=$RunRoot"

    Write-Host "[run-all] main_score_5k"
    & "$ScriptDir\run_main_score_5k.ps1" -Seeds $Seeds -Device $Device -PackageSize $PackageSize -OutputRoot $Main5kRoot -Python $Python
    if ($LASTEXITCODE -ne 0) { throw "main_score_5k failed with exit code $LASTEXITCODE" }

    Write-Host "[run-all] main_ablation_5k"
    & "$ScriptDir\run_main_ablation_5k.ps1" -Seeds $Seeds -Device $Device -PackageSize $PackageSize -OutputRoot $AblationRoot -Python $Python
    if ($LASTEXITCODE -ne 0) { throw "main_ablation_5k failed with exit code $LASTEXITCODE" }

    Write-Host "[run-all] ratio_downstream_5k"
    & "$ScriptDir\run_ratio_downstream_5k.ps1" -Seeds $Seeds -Ratios $Ratios -Device $Device -PackageSize $PackageSize -OutputRoot $RatioRoot -Python $Python
    if ($LASTEXITCODE -ne 0) { throw "ratio_downstream_5k failed with exit code $LASTEXITCODE" }

    Write-Host "[run-all] main_score_20k"
    & "$ScriptDir\run_main_score_20k.ps1" -Seeds $Seeds -Device $Device -PackageSize $PackageSize -OutputRoot $Main20kRoot -Python $Python
    if ($LASTEXITCODE -ne 0) { throw "main_score_20k failed with exit code $LASTEXITCODE" }

    Write-Host "[run-all] completed $(Get-Date -Format o)"
}
catch {
    Write-Host "[run-all] failed $(Get-Date -Format o)"
    Write-Host $_
    throw
}
finally {
    Stop-Transcript | Out-Null
}
