# Polynomial Regression from Scratch

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-orange)](https://numpy.org)
[![Documentation](https://img.shields.io/badge/Docs-PDF-brightgreen)](docs.pdf)

A complete implementation of polynomial regression with:
- Pure NumPy implementation (no ML library dependencies for core algorithm)
- Automatic PDF documentation generation
- Interactive visualization and model comparison
- Comprehensive evaluation metrics

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Complete Workflow](#complete-workflow)
  - [Documentation Generation](#documentation-generation)
- [Mathematical Background](#mathematical-background)
- [API Reference](#api-reference)
- [Contributing](#contributing)


## Features

### Core Functionality
✅ Polynomial regression via normal equations  
✅ Support for arbitrary polynomial degrees  
✅ Feature transformation and model training  
✅ Prediction and evaluation methods  

### Analysis Tools
📊 Comparison of multiple polynomial degrees  
📈 Visualization of fits and residuals  
📉 MSE/MAE/R² metrics for model evaluation  

### Documentation
📄 Automated PDF report generation  
📖 Mathematical theory explanation  
🧩 Code examples and API reference  

## Installation

1. Clone the repository:
```bash
git clone [(https://github.com/Karansehgal0611/PolynomialRegCustom.git)
cd PolynomialRegCustom
```
2.Install dependencies:
```bash
pip install -r requirements.txt
```
Requirements:
- Python 3.8+
- NumPy
- Matplotlib
- scikit-learn (for metrics)
- ReportLab (for PDF docs)

## Usage

### Basic Usage

```python
from polynomial_regression import PolynomialRegression
import numpy as np

# Generate sample data
X = np.linspace(-3, 3, 100)
y = 2 + 3*X - 1.5*X**2 + 0.5*X**3 + np.random.normal(0, 2, 100)

# Initialize and fit model
model = PolynomialRegression(degree=3)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Evaluate model
r2_score = model.score(X, y)
print(f"R² Score: {r2_score:.4f}")
```
### Complete Workflow

```python
from polynomial_regression import polynomial_regression_workflow

# Run complete analysis
results = polynomial_regression_workflow(
    X, y,
    degrees=[1, 2, 3, 4, 5],
    true_function=lambda x: 2 + 3*x - 1.5*x**2 + 0.5*x**3,
    title="Cubic Polynomial Example"
)

# Access results
print(f"Best polynomial degree: {results.best_degree}")
print("All metrics:", results.metrics)

# Display plots (blocks execution)
import matplotlib.pyplot as plt
plt.show()
```
### Documentation Generation

```python
from polynomial_regression import generate_polynomial_regression_documentation

# Generate comprehensive PDF docs
generate_polynomial_regression_documentation("polynomial_regression_docs.pdf")
```

## Mathematical Background

The polynomial regression model has the form:

$$
y = \beta_0 + \beta_1x + \beta_2x^2 + \cdots + \beta_nx^n + \varepsilon
$$

Where:
- $y$: Dependent variable (target)
- $x$: Independent variable (feature)
- $\beta_i$: Model coefficients
- $\varepsilon$: Error term

Coefficients are calculated using the normal equation:

$$
\beta = (X^T X)^{-1} X^T y
$$

## API Reference
### PolynomialRegression Class
```python
PolynomialRegression(degree=2)
``` 
### Polynomial_regression_workflow Function
```python
polynomial_regression_workflow(
    X: np.ndarray,
    y: np.ndarray,
    degrees: List[int] = [1, 2, 3, 5],
    test_size: float = 0.2,
    random_state: int = 42,
    true_function: Optional[Callable] = None,
    title: str = "Polynomial Regression Analysis"
) -> Dict[str, Any]
```

## Contributions
Contributions welcome! Please follow these steps:

- Fork the repository
- Create a feature branch (git checkout -b feature/your-feature)
- Commit your changes (git commit -am 'Add some feature')
- Push to the branch (git push origin feature/your-feature)
- Open a Pull Request

