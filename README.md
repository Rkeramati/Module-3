# MiniTorch Module 3 - Parallel and GPU Acceleration

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

**Documentation:** https://minitorch.github.io/

**Overview (Required reading):** https://minitorch.github.io/module3/module3/

## Overview

Module 3 focuses on **optimizing tensor operations** through parallel computing and GPU acceleration. You'll implement CPU parallel operations using Numba and GPU kernels using CUDA, achieving dramatic performance improvements over the sequential tensor backend from Module 2.

### Key Learning Goals
- **CPU Parallelization**: Implement parallel tensor operations with Numba
- **GPU Programming**: Write CUDA kernels for tensor operations
- **Performance Optimization**: Achieve significant speedup through hardware acceleration
- **Matrix Multiplication**: Optimize the most computationally intensive operations with operator fusion

## Tasks Overview

**Task 3.1**: CPU Parallel Operations
File to edit: `minitorch/fast_ops.py`
Feel free to use numpy functions like `np.array_equal()` and `np.zeros()`.

**Task 3.2**: CPU Matrix Multiplication
File to edit: `minitorch/fast_ops.py`
Implement optimized batched matrix multiplication with parallel outer loops.

**Task 3.3**: GPU Operations (requires GPU)
File to edit: `minitorch/cuda_ops.py`
Implement CUDA kernels for tensor map, zip, and reduce operations.

**Task 3.4**: GPU Matrix Multiplication (requires GPU)
File to edit: `minitorch/cuda_ops.py`
Implement CUDA matrix multiplication with shared memory optimization for maximum performance.

**Task 3.5**: Training (requires GPU)
File to edit: `project/run_fast_tensor.py`
Implement missing functions and train models on all datasets to demonstrate performance improvements.

## Documentation

- **[Installation Guide](installation.md)** - Setup instructions including GPU configuration
- **[Testing Guide](testing.md)** - How to run tests locally and handle GPU requirements

## GPU Setup

Follow this [link](https://colab.research.google.com/drive/1gyUFUrCXdlIBz9DYItH9YN3gQ2DvUMsI?usp=sharing). Go to the Colab file → save to drive, select runtime to T4 and follow instructions.

## Development Tools
## Code Quality
```bash
# Automatic style checking
pre-commit install
git commit -m "your changes"  # Runs style checks automatically

# Manual style checks
ruff check .      # Linting
ruff format .     # Formatting
pyright .         # Type checking
```

### Debugging
```bash
# Debug Numba JIT issues
NUMBA_DISABLE_JIT=1 pytest -m task3_1 -v

# Debug CUDA kernels
NUMBA_CUDA_DEBUG=1 pytest -m task3_3 -v

# Monitor GPU usage
nvidia-smi -l 1  # Update every second
```

### Training Commands

#### Local Environment
```bash
# CPU Backend
python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05

# GPU Backend  
python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
```

#### Google Colab (Recommended)
```bash
# GPU Backend examples
!cd $DIR; PYTHONPATH=/content/$DIR python3.11 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
!cd $DIR; PYTHONPATH=/content/$DIR python3.11 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
!cd $DIR; PYTHONPATH=/content/$DIR python3.11 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
```

### Student Results
**TODO: Add your training results here**

#### Simple Dataset
- CPU Backend: [Add time per epoch and accuracy]
- GPU Backend: [Add time per epoch and accuracy]

#### Split Dataset  
- CPU Backend: [Add time per epoch and accuracy]
- GPU Backend: [Add time per epoch and accuracy]

#### XOR Dataset
- CPU Backend: [Add time per epoch and accuracy] 
- GPU Backend: [Add time per epoch and accuracy]
