param(
    [string]$Rows = "5000",
    [string]$Dims = "64",
    [string]$Schemes = "baseline_random_noop,baseline_cosine_ctpt,baseline_uncertainty_poly4_ctpt,baseline_coreset_all_distances_ctpt,baseline_badge_components_poly4_ctpt,baseline_kmeans_all_distances_ctpt,baseline_typiclust_sqrt_poly4_ctpt,ours_krr_row_exp_poly4_ctpt,ours_krr_pkg_exp_poly4_ctpt",
    [int]$Repeats = 3,
    [int]$Warmups = 1,
    [int]$PackageSize = 2,
    [int]$StudentSummaryDim = 10,
    [int]$ThresholdParties = 3,
    [string]$Distro = "Ubuntu-20.04",
    [Parameter(Mandatory = $true)]
    [string]$SealDir,
    [int]$KrrLandmarks = 1000,
    [int]$CoresetReferences = 800,
    [int]$KmeansCenters = 20,
    [int]$TypiclustNeighbors = 8,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$BenchDir = Join-Path $RepoRoot "encrypted_benchmarks\krr_ckks_seal"
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $ScriptDir "runs\ckks_poly4_all_methods"
}

& "$BenchDir\run_ckks_seal_bench_wsl.ps1" `
    -Distro $Distro `
    -SealDir $SealDir `
    -OutputDir $OutputDir `
    -Rows $Rows `
    -Dims $Dims `
    -Schemes $Schemes `
    -Repeats $Repeats `
    -Warmups $Warmups `
    -PackageSize $PackageSize `
    -StudentSummaryDim $StudentSummaryDim `
    -ThresholdParties $ThresholdParties `
    -KrrLandmarks $KrrLandmarks `
    -CoresetReferences $CoresetReferences `
    -KmeansCenters $KmeansCenters `
    -TypiclustNeighbors $TypiclustNeighbors `
    -MeasureDecryptAll

Write-Host "[done] student full-flow CKKS results: $OutputDir\ckks_seal_summary.csv"
