# Contributing to Threatora 🛡️

Thank you for your interest in contributing to **Threatora**! We welcome community contributions, bug reports, feature enhancements, and benchmark extensions.

## Code of Conduct
By participating in this project, you agree to abide by the terms of our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. **Fork the Repository**:
   Click the **Fork** button at the top right of [Threatora repository](https://github.com/hasim2006/Threatora).

2. **Clone your fork**:
   ```bash
   git clone https://github.com/<your-username>/Threatora.git
   cd Threatora
   ```

3. **Set Up Development Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install pytest black flake8
   ```

4. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/awesome-detection-mechanism
   ```

## Development Guidelines

- **Code Style**: Follow PEP 8 guidelines. Run `black .` and `flake8` before submitting code.
- **Testing**: Write unit tests for new layers, architectures, or preprocessing routines under `tests/`.
- **Benchmarking**: If introducing architectural modifications, provide comparative evaluation on UNSW-NB15 or CSE-CIC-IDS2018 datasets.
- **Documentation**: Update documentation and docstrings explaining architectural nuances.

## Submitting a Pull Request (PR)

1. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "feat(capsnet): add multi-head capsule routing"
   ```
2. Push to your branch:
   ```bash
   git push origin feature/awesome-detection-mechanism
   ```
3. Open a Pull Request against `main`. Fill in the PR description template.

Thank you for helping make Threatora stronger!
