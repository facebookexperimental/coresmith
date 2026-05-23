# Contributing to coresmith

We welcome contributions to coresmith. By participating, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Install development dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e "orchestrator[dev]"
   ```
4. Make your changes
5. Run tests: `pytest orchestrator/tests/ -v`
6. Submit a pull request

## Development Guidelines

- Target Python 3.11+
- Use `ruff` for linting (`ruff check orchestrator/`)
- Write tests for new functionality
- Keep commits focused and well-described

## Reporting Issues

Use GitHub Issues to report bugs or request features. Include:
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Relevant log output

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
