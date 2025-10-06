# MiniTorch Module 3 - Parallel and GPU Acceleration

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

**Documentation:** https://minitorch.github.io/

**Overview (Required reading):** https://minitorch.github.io/module3.html

## Overview

Module 3 focuses on **optimizing tensor operations** through parallel computing and GPU acceleration. You'll implement CPU parallel operations using Numba and GPU kernels using CUDA, achieving dramatic performance improvements over the sequential tensor backend from Module 2.

### Key Learning Goals
- **CPU Parallelization**: Implement parallel tensor operations with Numba
- **GPU Programming**: Write CUDA kernels for tensor operations
- **Performance Optimization**: Achieve significant speedup through hardware acceleration
- **Matrix Multiplication**: Optimize the most computationally intensive operations
- **Backend Architecture**: Build multiple computational backends for flexible performance

## Tasks Overview

| Task    | Description 
|---------|-------------
| **3.1** | CPU Parallel Operations (`fast_ops.py`) 
| **3.2** | CPU Matrix Multiplication (`fast_ops.py`) 
| **3.3** | GPU Operations (`cuda_ops.py`)
| **3.4** | GPU Matrix Multiplication (`cuda_ops.py`)
| **3.5** | Performance Evaluation (`run_fast_tensor.py`)

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

## Important Notes

- **GPU Limitations**: Tasks 3.3 and 3.4 cannot run in GitHub CI due to hardware requirements
- **GPU Testing**: Use Google Colab (recommended) or local NVIDIA GPU for GPU tasks
- **Performance Critical**: Implementations must show measurable speedup over sequential versions
- **Memory Management**: Be careful with GPU memory allocation and deallocation
