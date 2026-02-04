# Contributing to Seismic Fractal Analysis (SFA)

Thank you for considering contributing to the SFA framework! We welcome contributions from the community to improve the software, documentation, and validation.

---

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on [GitHub Issues](https://github.com/FacundoFirmenich/SeismicFractalAnalysis/issues) with:
- **Title**: Brief description of the bug
- **Steps to reproduce**: Minimal code example
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: Python version, OS, SFA version

### Suggesting Enhancements

We appreciate suggestions for new features or improvements. Please open an issue with:
- **Title**: Brief description of the enhancement
- **Motivation**: Why this feature would be useful
- **Proposed implementation**: If you have ideas on how to implement it

### Pull Requests

We follow a standard fork-and-pull request workflow:

1. **Fork** the repository to your GitHub account
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/SeismicFractalAnalysis.git
   cd SeismicFractalAnalysis
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** and commit with descriptive messages:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```
5. **Push** your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a pull request** on the main repository

---

## Development Guidelines

### Code Style

- Follow **PEP 8** for Python code style
- Maximum line length: 100 characters
- Use type hints for function signatures
- Write docstrings for all public functions/classes (NumPy style)

Example:
```python
def compute_dimension(coords: np.ndarray, method: str = "gp") -> Tuple[float, float]:
    """
    Compute correlation dimension of earthquake catalog.
    
    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        Coordinates (lat, lon, depth) of N earthquakes
    method : str, default="gp"
        Method to use: "gp" (Grassberger-Procaccia) or "takens"
    
    Returns
    -------
    dimension : float
        Estimated correlation dimension
    sem : float
        Standard error of the mean (bootstrap uncertainty)
    
    Examples
    --------
    >>> coords = np.random.rand(1000, 3)
    >>> d2, d2_sem = compute_dimension(coords)
    >>> print(f"D₂ = {d2:.3f} ± {d2_sem:.3f}")
    """
    ...
```

### Testing

- All new features must include **unit tests** using pytest
- Aim for **95%+ code coverage**
- Run tests before submitting pull request:
  ```bash
  pytest tests/ -v --cov=sfa --cov-report=html
  ```

### Documentation

- Update docstrings for any modified functions
- Add examples to `notebooks/` if introducing major new features
- Update `README.md` if changing installation/usage instructions
- Update API documentation in `docs/` if adding new modules

---

## Areas Needing Help

We particularly welcome contributions in the following areas:

### 1. GPU Backend Testing
- **Task**: Validate CuPy/JAX GPU implementations with N>100k catalogs
- **Skills**: Python, GPU programming (CUDA), NumPy array manipulation
- **Benefit**: Enable real-time analysis of large global catalogs

### 2. Additional Validation Datasets
- **Task**: Apply SFA to new high-precision catalogs (e.g., European arrays, New Zealand GeoNet)
- **Skills**: Seismology domain knowledge, data acquisition
- **Benefit**: Cross-validate precision dependency hypothesis beyond USGS/Hi-Net

### 3. Documentation Translations
- **Languages needed**: Spanish, Japanese, Chinese, German
- **Task**: Translate README.md, tutorial notebooks, API docs
- **Skills**: Bilingual fluency + basic Python knowledge
- **Benefit**: Increase global accessibility

### 4. Performance Benchmarking
- **Task**: Systematically benchmark SFA vs ZMAP for various N (10²-10⁶)
- **Skills**: Python profiling, statistical analysis
- **Benefit**: Quantify scalability advantages for publication

### 5. Interactive Visualization
- **Task**: Develop Plotly/Dash dashboard for real-time catalog monitoring
- **Skills**: Plotly/Dash, web development basics
- **Benefit**: Enable operational use by seismologists without Python expertise

---

## Code of Conduct

We adhere to the **Contributor Covenant** code of conduct. In summary:

- Be respectful and inclusive
- Welcome diverse perspectives
- Focus on constructive criticism
- Assume good faith

Full text: https://www.contributor-covenant.org/version/2/1/code_of_conduct/

---

## Questions?

If you have questions about contributing, please:
- Open a GitHub Discussion: https://github.com/FacundoFirmenich/SeismicFractalAnalysis/discussions
- Email: f.firmenich@cedesur.org

---

## Attribution

Contributors will be acknowledged in:
- `AUTHORS.md` file
- Release notes
- Future scientific publications (if substantial contribution)

Thank you for helping improve SFA! 🌍🔬
