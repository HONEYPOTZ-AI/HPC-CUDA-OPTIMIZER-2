# HPC-CUDA-OPTIMIZER-2

# 1. Rename the file
mv hpc_cuda_optimizer_extract.txt hpc_cuda_optimizer_extract.py

# 2. Extract — creates hpc-cuda-optimizer/ in the current directory
python3 hpc_cuda_optimizer_extract.py

| Directory | Contents |
| --- | --- |
| ``src/`` | ``optimizer_main.cpp``, ``optimizer_kernels.cu``, ``optimizer_kernels.cuh`` |
| ``bench/`` | ``bench_reduce.cu``, ``bench_zero3.cpp``, ``bench_zero3_nccl.cpp``, ``bench_zero3_nccl_multi.cpp`` |
| ``slurm/`` | ``ci_bench.slurm``, ``nsys_profile.slurm``, ``zero3_bench.slurm``, ``zero3_nccl_bench.slurm``, ``zero3_nccl_multi_bench.slurm`` |
| ``scripts/`` | ``wait_for_job.sh``, ``build_and_submit.sh``, ``check_perf.py``, ``check_zero3_perf.py``, ``check_zero3_nccl_perf.py`` |
| ``configs/`` | ``optimizer_config.yaml``, ``slurm_defaults.yaml`` |
| ``tests/`` | ``test_optimizer.cpp`` + ``CMakeLists.txt`` |
| ``.github/workflows/`` | ``ci.yml`` (lint + checker self-tests), ``gpu_bench.yml`` (SLURM dispatch) |
| Root | ``CMakeLists.txt``, ``README.md``, ``LICENSE``, ``.gitignore`` |

module load cuda/12.4 openmpi/4.1.6-cuda12 nccl/2.20.5-cuda12
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;89;90"
make -j$(nproc)
ctest --output-on-failure   # runs unit tests
