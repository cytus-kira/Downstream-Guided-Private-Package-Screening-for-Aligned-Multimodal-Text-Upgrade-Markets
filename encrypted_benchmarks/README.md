# Encrypted CPU/GPU Benchmarks

`seal_cpu` contains the active CPU ciphertext-compute, full-ledger, and packing-ablation programs. It requires Microsoft SEAL 4.x.

`seal_gpu` contains the CUDA-linked experimental comparison program. The current source uses the Microsoft SEAL API and links the CUDA runtime, but it does not implement custom CUDA kernels. Treat it as the retained GPU test variant, not as evidence that Microsoft SEAL execution is GPU-native.

Third-party Microsoft SEAL/OpenFHE source trees, build directories, binaries, logs, and generated CSV outputs are intentionally excluded.
