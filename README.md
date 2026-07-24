# SecGuardFuse 实验代码复现包

这是从 `ball_market` 工作区整理出的干净、可移植源码包。它保留当前论文实验实际调用的实现、编排脚本、CKKS 基准、两方可验证释放原型，以及约 1.5 MB 的机器可读证据快照；不包含特征缓存、市场 NPZ、模型权重、私钥、密文转录、编译产物或历史探索脚本。

## 目录

- `hm_tdsc_nonpackage_ablations.py`：数据、模型、选择与评估公共实现。
- `paper/run_quick_downstream_collapse_experiment.py`：下游教师、四次 KRR、原始行/买方同步打包的锁定实现。
- `paper/main_text_experiments/`：主实验、消融、预算曲线、包大小、近似保真、表图汇总入口。
- `encrypted_benchmarks/`：Microsoft SEAL CPU、CUDA-linked 测试版和完整 CKKS KRR 基准。
- `run_two_party_release.py` 与 `vtd/`：Lattigo 两方 CKKS、gnark Groth16 关系证明、购买后审计。
- `evidence/`：RQ1–RQ5 及下游训练的精简 CSV/JSON 快照。
- `EXPERIMENT_SUMMARY.md`：实验协议、结果、限制与证据索引。
- `CODE_INVENTORY.md`：现行调用链和纳入/排除范围。
- `verify_release.py`：无数据静态验证入口。

## 环境

Python 3.10+：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

正式 GPU 实验需要与本机 CUDA 匹配的 PyTorch。两方 VTD 需要 Go 1.25.x。SEAL 基准需要 Microsoft SEAL 4.x、CMake；包内 WSL 包装脚本接受外部 `SEAL_DIR`，不绑定作者机器路径。

特征缓存需由使用者放在：

```text
feature_cache/<dataset>/<encoder>/base
```

大缓存未随包发布；没有缓存时仍可完成静态验证和命令展开，但不能重新训练或重建市场。

## 先验证包

```powershell
python verify_release.py
python paper\main_text_experiments\run_main_text_experiments.py `
  --preset main_score_5k --seeds 42 --device cpu --dry-run
```

有 Go 时可追加：

```powershell
python verify_release.py --with-go
```

## 主要复现入口

从包根目录运行：

```powershell
.\paper\main_text_experiments\run_main_score_5k.ps1 -Seeds "42,43,44" -Device cuda
.\paper\main_text_experiments\run_main_ablation_5k.ps1 -Seeds "42,43,44" -Device cuda
.\paper\main_text_experiments\run_ratio_downstream_5k.ps1 -Seeds "42,43,44" -Device cuda
```

也可以直接使用 Python 编排器；命令与输出字段详见 `paper/main_text_experiments/README.md`。

CKKS 全流程基准：

```powershell
.\paper\main_text_experiments\run_ckks_student_fullflow_comm.ps1 `
  -SealDir C:\path\to\SEAL\build\cmake `
  -Distro Ubuntu-20.04
```

两方可验证释放的烟雾测试：

```powershell
python run_two_party_release.py --scores 1 --seed 42 --preset smoke --landmarks 8 --repeats 1
```

## 证据边界

`evidence/` 是已有运行的精简快照，不等于完整原始数据归档。主选择实验为 3 数据集 × 7 市场 × 3 种子（63 个 case）；CKKS 表是特定硬件上的微基准；VTD 是两方研究原型。具体负结果和安全边界见 `EXPERIMENT_SUMMARY.md` 与 `vtd/IMPLEMENTATION_AUDIT.md`。
