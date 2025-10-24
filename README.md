# MiniTorch Module 3 - Parallel and GPU Acceleration

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

**Documentation:** https://minitorch.github.io/

**Overview (Required reading):** https://minitorch.github.io/module3.html

## Overview

Module 3 focuses on **optimizing tensor operations** through parallel computing and GPU acceleration. You'll implement CPU parallel operations using Numba and GPU kernels using CUDA, achieving dramatic performance improvements over the sequential tensor backend from Module 2.

## Tasks Overview

**Task 3.1**: CPU Parallel Operations
File to edit: `minitorch/fast_ops.py`
Feel free to use numpy functions like `np.array_equal()` and `np.zeros()`.

**Task 3.2**: CPU Matrix Multiplication
File to edit: `minitorch/fast_ops.py`
Implement optimized batched matrix multiplication with parallel outer loops.

**Task 3.3**: GPU Operations
File to edit: `minitorch/cuda_ops.py`
Implement CUDA kernels for tensor map, zip, and reduce operations.

**Task 3.4**: GPU Matrix Multiplication
File to edit: `minitorch/cuda_ops.py`
Implement CUDA matrix multiplication with shared memory optimization for maximum performance.

**Task 3.5**: Training
File to edit: `project/run_fast_tensor.py`
Implement missing functions and train models on all datasets to demonstrate performance improvements.

## Documentation

- **[Installation Guide](installation.md)** - Setup instructions including GPU configuration
- **[Testing Guide](testing.md)** - How to run tests locally and handle GPU requirements

## Quick Start

### 1. Environment Setup
```bash
# Clone and navigate to your assignment
git clone <your-assignment-repo>
cd <assignment-directory>

# Create virtual environment (recommended)
conda create --name minitorch python
conda activate minitorch

# Install dependencies
pip install -e ".[dev,extra]"
```

### 2. Sync Previous Module Files
```bash
# Sync required files from your Module 2 solution
python sync_previous_module.py <path-to-module-2> .

# Example:
python sync_previous_module.py ../Module-2 .
```

### 3. Run Tests
```bash
# CPU tasks (run anywhere)
pytest -m task3_1  # CPU parallel operations
pytest -m task3_2  # CPU matrix multiplication

# GPU tasks (require CUDA-compatible GPU)
pytest -m task3_3  # GPU operations
pytest -m task3_4  # GPU matrix multiplication

# Style checks
pre-commit run --all-files
```

## GPU Setup

### Option 1: Google Colab (Recommended)
Most students should use Google Colab for GPU tasks:
https://colab.research.google.com/drive/1unW9QzB1eFDZ86hGJlIYACXvivhJH3YT
1. Upload assignment files to Colab
2. Change runtime to GPU (Runtime → Change runtime type → GPU)
3. Install packages:
   ```python
   !pip install -e ".[dev,extra]"
   !python -c "import numba.cuda; print('CUDA available:', numba.cuda.is_available())"
   ```

### Option 2: Local GPU (If you have NVIDIA GPU)
For students with NVIDIA GPUs and CUDA-compatible hardware:

```bash
# Install CUDA toolkit
# Visit: https://developer.nvidia.com/cuda-downloads

# Install GPU packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numba[cuda]

# Verify GPU support
python -c "import numba.cuda; print('CUDA available:', numba.cuda.is_available())"
```

## Testing Strategy

### CI/CD (GitHub Actions)
- **Task 3.1**: CPU parallel operations
- **Task 3.2**: CPU matrix multiplication  
- **Style Check**: Code quality and formatting

### GPU Testing (Colab/Local GPU)
- **Task 3.3**: GPU operations (use Colab or local NVIDIA GPU)
- **Task 3.4**: GPU matrix multiplication (use Colab or local NVIDIA GPU)

### Performance Validation
```bash
# Compare backend performance
python project/run_fast_tensor.py    # Optimized backends
python project/run_tensor.py         # Basic tensor backend
python project/run_scalar.py         # Scalar baseline
```

## Development Tools

### Code Quality
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

## Implementation Focus

### Task 3.1 & 3.2 (CPU Optimization)
- Implement `tensor_map`, `tensor_zip`, `tensor_reduce` with Numba parallel loops
- Optimize matrix multiplication with efficient loop ordering
- Focus on cache locality and parallel execution patterns

### Task 3.3 & 3.4 (GPU Acceleration)  
- Write CUDA kernels for element-wise operations
- Implement efficient GPU matrix multiplication with shared memory
- Optimize thread block organization and memory coalescing

## Task 3.5 Training Results

### Performance Targets
- **CPU Backend**: Below 2 seconds per epoch
- **GPU Backend**: Below 1 second per epoch (on standard Colab GPU)

### Training Commands
```bash
# CPU Backend
python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET <dataset> --RATE 0.05

# GPU Backend  
python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET <dataset> --RATE 0.05
```

### Student Results
**TODO: Add your training results here**
