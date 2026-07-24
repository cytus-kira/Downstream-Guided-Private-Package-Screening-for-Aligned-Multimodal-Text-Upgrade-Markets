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
    $OutputRoot = "$ScriptDir\runs\main_ablation_5k"
}
$Methods = @(
    "ours_downstream_direct",
    "ours_influence_only",
    "ours_loss_reduction_only",
    "ours_task_operator",
    "ours_krr_influence_only",
    "ours_krr_loss_reduction_only",
    "ours_kernel_ridge_student",
    "ours_sample_package_direct",
    "ours_sample_package_influence_only",
    "ours_sample_package_loss_reduction_only",
    "ours_sample_package_task_operator",
    "ours_sample_package_krr_influence_only",
    "ours_sample_package_krr_loss_reduction_only",
    "ours_sample_package_krr"
) -join ","

& $Python "$ScriptDir\run_main_text_experiments.py" `
    --preset main_score_5k `
    --seeds $Seeds `
    --device $Device `
    --package-size $PackageSize `
    --output-root $OutputRoot `
    --methods $Methods `
    --score-only
