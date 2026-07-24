param(
    [string]$Distro = "Ubuntu-20.04",
    [Parameter(Mandatory = $true)]
    [string]$SealDir,
    [string]$OutputDir = (Join-Path $PSScriptRoot "results"),
    [string]$Rows = "100,500,1000,5000,10000",
    [string]$Dims = "16,32,64",
    [string]$Schemes = "all",
    [int]$Repeats = 5,
    [int]$Warmups = 0,
    [int]$PolyModulusDegree = 8192,
    [string]$CoeffModulusBits = "45,32,32,32,32,45",
    [int]$ScaleBits = 32,
    [int]$PackageSize = 4,
    [int]$StudentSummaryDim = 10,
    [int]$Seed = 42,
    [int]$ThresholdParties = 1,
    [int]$KrrLandmarks = 1000,
    [int]$CoresetReferences = 800,
    [int]$KmeansCenters = 20,
    [int]$TypiclustNeighbors = 8,
    [switch]$MeasureDecryptAll,
    [switch]$Validate
)

$ErrorActionPreference = "Stop"

function Convert-ToWslPath {
    param([string]$Path)
    $FullPath = [System.IO.Path]::GetFullPath($Path)
    $Drive = $FullPath.Substring(0, 1).ToLowerInvariant()
    $Rest = $FullPath.Substring(2).Replace("\", "/")
    return "/mnt/$Drive$Rest"
}

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$WslProjectDir = Convert-ToWslPath $ProjectDir
$WslSealDir = Convert-ToWslPath $SealDir
$WslOutputDir = Convert-ToWslPath $OutputDir
$WslOutCsv = "$WslOutputDir/ckks_seal_results.csv"

$ValidateArg = ""
if ($Validate) {
    $ValidateArg = "--validate"
}
$DecryptArg = ""
if ($MeasureDecryptAll) {
    $DecryptArg = "--measure-decrypt-all"
}

$cmd = @"
set -e
cd '$WslProjectDir'
cmake -S . -B build -DSEAL_DIR='$WslSealDir' -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
mkdir -p '$WslOutputDir'
./build/ckks_seal_bench \
  --rows '$Rows' \
  --dims '$Dims' \
  --schemes '$Schemes' \
  --repeats '$Repeats' \
  --warmups '$Warmups' \
  --poly '$PolyModulusDegree' \
  --coeff-bits '$CoeffModulusBits' \
  --scale-bits '$ScaleBits' \
  --package-size '$PackageSize' \
  --student-summary-dim '$StudentSummaryDim' \
  --seed '$Seed' \
  --threshold-parties '$ThresholdParties' \
  --krr-landmarks '$KrrLandmarks' \
  --coreset-references '$CoresetReferences' \
  --kmeans-centers '$KmeansCenters' \
  --typiclust-neighbors '$TypiclustNeighbors' \
  --out '$WslOutCsv' \
  $DecryptArg \
  $ValidateArg
"@

wsl.exe -d $Distro -- bash -lc $cmd

$PythonExe = "python"
& $PythonExe "$ProjectDir\summarize_ckks_results.py" --input "$OutputDir\ckks_seal_results.csv" --out-dir "$OutputDir"
Write-Host "[done] results: $OutputDir\ckks_seal_results.csv"
Write-Host "[done] summary: $OutputDir\ckks_seal_summary.csv"
