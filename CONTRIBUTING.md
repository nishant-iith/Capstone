# Contributing to Virtual H&E Stain Generator

Thank you for your interest in contributing! This guide explains how to help improve the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/capstone.git
   cd Capstone
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install in development mode**:
   ```bash
   pip install -r requirements.txt
   ```

## Making Changes

### Code Style
- Use clear, descriptive variable names
- Add comments for non-obvious logic
- Keep functions focused and under 50 lines when possible
- Follow PEP 8 naming conventions

### Testing
- Test locally before submitting:
  ```bash
  python train_v7.py --max_epochs 2  # Quick test
  python infer_v7_tta.py
  ```
- Verify no errors in console output

### Commit Messages
- Use clear, concise messages
- Start with a verb: "Add", "Fix", "Update", "Remove"
- Example: "Add ensemble inference with 3 checkpoints"

## Types of Contributions

### Bug Fixes
- Describe the bug and how you fixed it
- Include steps to reproduce
- Add before/after metrics if applicable

### New Features
- Propose the feature in an issue first
- Discuss approach with maintainers
- Keep scope focused (avoid adding 10 features at once)

### Documentation
- Improve README clarity
- Add usage examples
- Fix typos or unclear explanations
- Document any new features

### Model Improvements
- Test thoroughly on validation set
- Report SSIM and PSNR metrics
- Explain why the change helps
- Document any new hyperparameters

## Submitting Your Work

1. **Push to your fork**:
   ```bash
   git push origin feature-branch
   ```
2. **Create a pull request** on GitHub
3. **Fill out the PR template** with:
   - What changes you made
   - Why you made them
   - Any new metrics or results
   - Related issues (if any)

## Project Status

**Final Model**: v7 (SSIM 0.2696 with TTA)

The model is production-ready. Contributions that improve upon v7's baseline are welcome. Future directions:
- Multi-slide training (dataset expansion)
- Improved stain normalization
- Faster inference strategies

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Be descriptive: "Model crashes on 256x256 input" vs "Something wrong"

## License

By contributing, you agree your code is licensed under MIT (see LICENSE file).

---

**Thank you for contributing!**
