## Testing Your Implementation

### Running Tests

This project uses pytest for testing. Tests are organized by task:

```bash
# Run all tests for a specific task
pytest -m task3_1  # CPU parallel operations
pytest -m task3_2  # CPU matrix multiplication
pytest -m task3_3  # GPU operations (requires CUDA)
pytest -m task3_4  # GPU matrix multiplication (requires CUDA)

# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run a specific test file
pytest tests/test_tensor_general.py  # All optimized tensor tests

# Run a specific test function
pytest tests/test_tensor_general.py::test_one_args -k "fast"
pytest tests/test_tensor_general.py::test_matrix_multiply
```

### GPU Testing Strategy

**CI Limitations:**
- GitHub Actions CI only runs tasks 3.1 and 3.2 (CPU only)
- Tasks 3.3 and 3.4 require local GPU or Google Colab

**Option 1: Google Colab Testing (Recommended):**
```python
# In Colab notebook
!pip install -e ".[dev,extra]"
!python -m pytest -m task3_3 -v
!python -m pytest -m task3_4 -v
!python -c "import numba.cuda; print('CUDA available:', numba.cuda.is_available())"
```

**Option 2: Local GPU Testing (If you have NVIDIA GPU):**
```bash
# Verify CUDA is available
python -c "import numba.cuda; print('CUDA available:', numba.cuda.is_available())"

# Test GPU tasks locally
pytest -m task3_3  # GPU operations
pytest -m task3_4  # GPU matrix multiplication

# Debug GPU issues
NUMBA_DISABLE_JIT=1 pytest -m task3_3 -v  # Disable JIT for debugging
```

### Style and Code Quality Checks

This project enforces code style and quality using several tools:

```bash
# Run all pre-commit hooks (recommended)
pre-commit run --all-files

# Individual style checks:
ruff check .                 # Linting (style, imports, docstrings)
ruff format .               # Code formatting
pyright .                   # Type checking
```

### Task 3.5 - Performance Evaluation

**Training Scripts:**
```bash
# Run optimized training (CPU parallel)
python project/run_fast_tensor.py

# Compare with previous implementations
python project/run_tensor.py     # Basic tensor implementation
python project/run_scalar.py     # Scalar implementation
```

### Parallel Diagnostics (Tasks 3.1 & 3.2)

**Running Parallel Check:**
```bash
# Verify your parallel implementations
python project/parallel_check.py
```

**Expected Output for Task 3.1:**
- **MAP**: Should show parallel loops for both fast path and general case with allocation hoisting for `np.zeros()` calls
- **ZIP**: Should show parallel loops for both fast path and general case with optimized memory allocations
- **REDUCE**: Should show main parallel loop with proper allocation hoisting

**Expected Output for Task 3.2:**
- **MATRIX MULTIPLY**: Should show nested parallel loops for batch and row dimensions with no allocation hoisting (since no index buffers are used)

**Key Success Indicators:**
- Parallel loops detected with `prange()`
- Memory allocations hoisted out of parallel regions
- Loop optimizations applied by Numba
- No unexpected function calls in critical paths

### Pre-commit Hooks (Automatic Style Checking)

The project uses pre-commit hooks that run automatically before each commit:

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install

# Now style checks run automatically on every commit
git commit -m "your message"  # Will run style checks first
```

### Debugging Tools

**Numba Debugging:**
```bash
# Disable JIT compilation for debugging
NUMBA_DISABLE_JIT=1 pytest -m task3_1 -v

# Enable Numba debugging output
NUMBA_DEBUG=1 python project/run_fast_tensor.py
```

**CUDA Debugging:**
```bash
# Check CUDA device properties
python -c "import numba.cuda; print(numba.cuda.gpus)"

# Monitor GPU memory usage
nvidia-smi -l 1  # Update every second

# Debug CUDA kernel launches
NUMBA_CUDA_DEBUG=1 python -m pytest -m task3_3 -v
```

**Performance Profiling:**
```bash
# Time specific operations
python -c "
import time
import minitorch
backend = minitorch.TensorBackend(minitorch.FastOps)
# Time your operations here
"

# Profile memory usage
python -m memory_profiler project/run_fast_tensor.py
```