# Downstream-Guided Private Package Screening for Aligned Multimodal Text Enrichment

## 1. Requirements

Install the components needed for the experiment you want to run:

- Python 3.10 or later
- PyTorch with a CUDA build for GPU experiments
- Go 1.25.x for the two-party verifiable threshold-release experiments
- Microsoft SEAL 4.1.x, CMake, and WSL for the CKKS benchmarks

The main selection and ablation experiments only require the Python environment and feature caches.

## 2. Python environment

Clone the repository and create a virtual environment:

```powershell
git clone https://github.com/cytus-kira/Downstream-Guided-Private-Package-Screening-for-Aligned-Multimodal-Text-Upgrade-Markets.git
Set-Location Downstream-Guided-Private-Package-Screening-for-Aligned-Multimodal-Text-Upgrade-Markets

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For formal GPU runs, install the PyTorch build that matches the local CUDA version before installing the remaining requirements.

## 3. Feature-cache layout

Place the precomputed features under the repository root:

```text
feature_cache/
├── hateful_memes/
│   └── clip_vit_base_patch32/base/
├── hatespeech/
│   └── clip_vit_base_patch32/base/
└── mscoco/
    └── clip_vit_base_patch32/base/
```

Each `base` directory must contain:

```text
train_img.npy
train_txt.npy
train_y.npy
test_img.npy
test_txt.npy
test_y.npy
```

## 4. Verify the environment

Run the static checks:

```powershell
python verify_release.py
```

Include the Go tests when Go is available on `PATH`:

```powershell
python verify_release.py --with-go
```

Inspect the main experiment command without starting training:

```powershell
python paper\main_text_experiments\run_main_text_experiments.py `
  --preset main_score_5k `
  --seeds 42 `
  --device cpu `
  --output-root .\outputs\dry_run `
  --dry-run
```

## 5. Run the main experiments

Run all commands from the repository root.

### Main 5k experiment

```powershell
.\paper\main_text_experiments\run_main_score_5k.ps1 `
  -Seeds "42,43,44" `
  -Device cuda `
  -PackageSize 2
```

### Main 20k experiment

```powershell
.\paper\main_text_experiments\run_main_score_20k.ps1 `
  -Seeds "42,43,44" `
  -Device cuda `
  -PackageSize 2
```

### Component ablation

```powershell
.\paper\main_text_experiments\run_main_ablation_5k.ps1 `
  -Seeds "42,43,44" `
  -Device cuda `
  -PackageSize 2
```

### Downstream purchase-ratio experiment

```powershell
.\paper\main_text_experiments\run_ratio_downstream_5k.ps1 `
  -Seeds "42,43,44" `
  -Ratios "0.0025,0.005,0.01,0.02" `
  -Device cuda `
  -PackageSize 2
```

### Package-size sweep

```powershell
python paper\main_text_experiments\run_package_size_sweep.py `
  --package-sizes "2,4,6,8" `
  --seeds "42,43,44" `
  --preset main_score_5k `
  --device cuda `
  --market-size 4992 `
  --purchase-total 48
```

Use `cpu` instead of `cuda` for small local checks. Formal experiment settings should use the same seeds, datasets, profiles, budgets, and package sizes across compared methods.

## 6. Run the CKKS benchmark

Install Microsoft SEAL 4.1.x and build its CMake package first. Then pass the SEAL CMake directory to the benchmark wrapper:

```powershell
.\paper\main_text_experiments\run_ckks_student_fullflow_comm.ps1 `
  -SealDir C:\path\to\SEAL\build\cmake `
  -Distro Ubuntu-20.04 `
  -Rows "5000" `
  -Dims "64" `
  -Repeats 3 `
  -Warmups 1 `
  -PackageSize 2
```

Change `-Distro` to the installed WSL distribution name when necessary:

```powershell
wsl.exe -l -q
```

## 7. Run the two-party release prototype

Ensure Go 1.25.x is available:

```powershell
go version
```

Run a small smoke test:

```powershell
python run_two_party_release.py `
  --scores 1 `
  --seed 42 `
  --preset smoke `
  --landmarks 8 `
  --repeats 1 `
  --run-root vtd\runs\smoke_seed42
```

Run the production-parameter configuration:

```powershell
python run_two_party_release.py `
  --scores 1 `
  --seed 42 `
  --preset production `
  --landmarks 1000 `
  --repeats 1 `
  --run-root vtd\runs\production_seed42
```

## 8. Output locations

- Main and ablation runs: `paper/main_text_experiments/runs/`
- Package-size summaries: `paper/main_text_experiments/paper_ready/`
- CKKS results: `paper/main_text_experiments/runs/ckks_poly4_all_methods/`
- Two-party release runs: `vtd/runs/`

Each main experiment run writes a manifest, per-seed outputs, and merged summaries under its output directory.
