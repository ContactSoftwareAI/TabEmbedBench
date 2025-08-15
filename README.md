# TabEmbedBench

TabEmbedBench is a benchmarking framework designed for evaluating tabular data embedding methods. 
The project includes wrappers for some Tabular Foundation models to extract their embeddings.

## Prerequisites

- Python 3.12 or higher
- uv package manager

## Setting up UV

UV is a fast Python package manager and project manager. 
If you don't have UV installed, you can install it using one of the following methods:

### Option 1: Using pip
```bash
pip install uv
```
### Option 2: Using the installer script (recommended)
``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
### Option 3: Using Homebrew (macOS/Linux)
``` bash
brew install uv
```
For more installation options, visit the [official UV documentation](https://docs.astral.sh/uv/getting-started/installation/).
## Installation
### Development Installation
1. **Clone the repository:**
``` bash
   git clone <repository-url>
   cd tabembedbench
```
1. **Install the package in development mode:**
``` bash
   uv sync --dev
```
This command will:
- Create a virtual environment if one doesn't exist
- Install all dependencies specified in `pyproject.toml`
- Install the package in editable mode for development

1. **Activate the virtual environment:**
``` bash
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate     # On Windows
```
Alternatively, you can run commands directly with UV without activating:
``` bash
   uv run python your_script.py
```
## GPU Support
The project includes PyTorch with CUDA 12.1 support for non-Darwin (non-macOS) systems. 
On macOS, the CPU version will be installed automatically.
## Development Tools
The project includes several development tools configured in : `pyproject.toml`
- **Ruff**: For code linting and formatting
``` bash
  uv run ruff check .      # Check for issues
  uv run ruff format .     # Format code
```

## Project Structure
``` 
tabembedbench/
├── src/                 # Source code
├── pyproject.toml      # Project configuration
└── README.md          # This file
```


