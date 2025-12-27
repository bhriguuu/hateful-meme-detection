# Contributing to Hateful Meme Detection

First off, thank you for considering contributing to this project! 🎉

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) CUDA-capable GPU for training

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/hateful-meme-detection.git
   cd hateful-meme-detection
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/hateful-meme-detection.git
   ```

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check existing issues. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the behavior
- Expected behavior
- Screenshots (if applicable)
- Your environment (OS, Python version, GPU, etc.)

### Suggesting Features

Feature requests are welcome! Please provide:

- A clear description of the feature
- The problem it solves
- Possible implementation approach

### Contributing Code

1. Check open issues for things to work on
2. Comment on an issue to let others know you're working on it
3. Create a branch for your work
4. Submit a pull request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py -v
```

### Code Formatting

```bash
# Format code with Black
black src/ discord_bot/ tests/

# Sort imports
isort src/ discord_bot/ tests/

# Check style with flake8
flake8 src/ discord_bot/ tests/
```

## Pull Request Process

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Test your changes**:
   ```bash
   pytest tests/ -v
   ```

4. **Commit your changes** with clear messages:
   ```bash
   git commit -m "Add: Brief description of what you added"
   git commit -m "Fix: Brief description of what you fixed"
   git commit -m "Update: Brief description of what you updated"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub:
   - Use a clear title and description
   - Reference any related issues
   - Include screenshots for UI changes

7. **Address review feedback** if requested

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use type hints for function arguments and return values
- Maximum line length: 100 characters
- Use docstrings for all public functions/classes

### Example

```python
def predict_meme(
    image_path: str,
    text: str,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Predict if a meme is hateful.
    
    Args:
        image_path: Path to the meme image.
        text: Text content of the meme.
        threshold: Classification threshold.
        
    Returns:
        Dictionary containing prediction results.
        
    Raises:
        FileNotFoundError: If image_path doesn't exist.
    """
    ...
```

### Commit Messages

Use clear, descriptive commit messages:

- `Add: [feature description]` - For new features
- `Fix: [bug description]` - For bug fixes
- `Update: [what was updated]` - For updates/improvements
- `Docs: [documentation change]` - For documentation changes
- `Test: [test description]` - For test additions/changes
- `Refactor: [refactor description]` - For code refactoring

### Documentation

- Update README.md if you change functionality
- Add docstrings for new functions/classes
- Update type hints

## Questions?

Feel free to open an issue with the `question` label if you have any questions about contributing.

---

Thank you for contributing! 🙏
