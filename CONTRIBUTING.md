# Contributing to HyDePh

Thank you for your interest in contributing to HyDePh! We welcome contributions ranging from bug fixes and documentation improvements to new physical algorithms and benchmark systems.

## Development Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gsurfer04/HyDePh.git
   cd HyDePh
   ```

2. **Create and activate a virtual or conda environment (Python >= 3.9):**
   ```bash
   conda create -n hydeph-dev python=3.11 -y
   conda activate hydeph-dev
   ```

3. **Install HyDePh in editable mode with development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

## Code Quality & Style Guidelines

- **Formatting & Linting:**
  - Code formatting is managed via `black`:
    ```bash
    black hydeph/ tests/ examples/
    ```
  - Fast linting and import sorting via `ruff`:
    ```bash
    ruff check hydeph/ tests/
    ```
  - Type checking via `mypy`:
    ```bash
    mypy hydeph/
    ```
- **Logging vs. Standard Output:**
  - Use `logger = logging.getLogger(__name__)` rather than `print()`.
- **Physical Quantities & Constants:**
  - Express energies in eV or meV, distances in Å, and times in fs.
  - Reference module-level physical constants (`HBAR_EV_FS`, `KB_EV_K`) rather than magic numerical literals.
- **Cross-Platform Compatibility:**
  - Always use `os.path.join()` or forward slashes for filesystem paths.
  - Do not hardcode Linux-specific paths; ensure Windows (MS-MPI) and Linux (Open MPI) execution remain supported.

## Running the Test Suite

Run the full automated test suite using `pytest`:
```bash
pytest tests/ -v
```

To run a specific test module:
```bash
pytest tests/test_components.py -v
```

## Submitting Pull Requests

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-enhancement
   ```
2. Commit your changes with descriptive messages:
   ```bash
   git commit -m "feat(lineshape): add custom smearing function"
   ```
3. Ensure the test suite passes locally with 100% success.
4. Update `CHANGELOG.md` under the `[Unreleased]` section.
5. Push to your fork and open a Pull Request against the `main` branch.
