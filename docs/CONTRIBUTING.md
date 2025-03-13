# Contributing to Chitu

First of all, thanks for taking the time to contribute!

We welcome contributions to the Chitu project! This document provides guidelines and instructions for contributing.

## Code Style

- Use `black` to format your Python code
- Use descriptive variable names
- Add docstrings for all public functions and classes
- Keep functions focused and single-purpose
- Add type hints where possible

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## Performance Testing

When submitting optimizations:

1. Run the standard benchmark suite in `benchmarks/`
2. Include before/after performance metrics
3. Document any hardware/software requirements


## Testing

- Unit tests: test/pytest contains all the unit tests. Run it with pytest test/ptest. Please ensure all tests pass before submitting a PR.
- Integrated tests: Inference of full models can be tested by test/single_req_test.py. Since many of the test cases require a dedicated hardware, currently we perform the tests only in Qingcheng.AI's private environment. Qingcheng.AI employees will help you validate and improve your PR through the tests.
- Performance tests for optimizatinos are welcome but not required.