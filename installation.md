# MiniTorch Module 3 Installation

MiniTorch requires Python 3.8 or higher. To check your version of Python, run:

```bash
>>> python --version
```

We recommend creating a global MiniTorch workspace directory that you will use
for all modules:

```bash
>>> mkdir workspace; cd workspace
```

## Environment Setup

We highly recommend setting up a *virtual environment*. The virtual environment lets you install packages that are only used for your assignments and do not impact the rest of the system.

**Option 1: Anaconda (Recommended)**
```bash
>>> conda create --name minitorch python    # Run only once
>>> conda activate minitorch
>>> conda install llvmlite                  # For optimization
```

**Option 2: Venv**
```bash
>>> python -m venv venv          # Run only once
>>> source venv/bin/activate
```

The first line should be run only once, whereas the second needs to be run whenever you open a new terminal to get started for the class. You can tell if it works by checking if your terminal starts with `(minitorch)` or `(venv)`.

## Getting the Code

Each assignment is distributed through a Git repo. Once you accept the assignment from GitHub Classroom, a personal repository under Cornell-Tech-ML will be created for you. You can then clone this repository to start working on your assignment.

```bash
>>> git clone {{ASSIGNMENT}}
>>> cd {{ASSIGNMENT}}
```

## Installation

Install all packages in your virtual environment:

```bash
>>> python -m pip install -e ".[dev,extra]"
```

## Syncing Previous Module Files

Module 3 requires files from Module 0, Module 1, and Module 2. Sync them using:

```bash
>>> python sync_previous_module.py <path-to-module-2> <path-to-current-module>
```

Example:
```bash
>>> python sync_previous_module.py ../Module-2 .
```

Replace `<path-to-module-2>` with the path to your Module 2 directory and `<path-to-current-module>` with `.` for the current directory.

This will copy the following required files:
- `minitorch/tensor_data.py`
- `minitorch/tensor_functions.py`
- `minitorch/tensor_ops.py`
- `minitorch/operators.py`
- `minitorch/scalar.py`
- `minitorch/scalar_functions.py`
- `minitorch/module.py`
- `minitorch/autodiff.py`
- `minitorch/tensor.py`
- `minitorch/datasets.py`
- `minitorch/testing.py`
- `minitorch/optim.py`
- `project/run_manual.py`
- `project/run_scalar.py`
- `project/run_tensor.py`

## GPU Setup (Required for Tasks 3.3 and 3.4)

Tasks 3.3 and 3.4 require CUDA and should be completed on **Google Colab** with GPU runtime.

Follow the instructions provided in the Colab notebook for GPU setup and testing.

### Option 2: Local GPU Setup (If you have NVIDIA GPU)

For students with NVIDIA GPUs and CUDA-compatible hardware:

1. **Install CUDA Toolkit**
   ```bash
   # Visit: https://developer.nvidia.com/cuda-downloads
   # Follow instructions for your OS
   ```

2. **Verify CUDA Installation**
   ```bash
   >>> nvcc --version
   >>> nvidia-smi
   ```

3. **Install GPU-compatible packages**
   ```bash
   >>> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   >>> pip install numba[cuda]
   ```

## Verification

Make sure everything is installed by running:

```bash
>>> python -c "import minitorch; print('Success!')"
```

Verify that the tensor functionality is available:

```bash
>>> python -c "from minitorch import tensor; print('Module 3 ready!')"
```

Check if CUDA support is available (for GPU tasks):

```bash
>>> python -c "import numba.cuda; print('CUDA available:', numba.cuda.is_available())"
```

You're ready to start Module 3!