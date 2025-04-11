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
- [Examples](#examples)
- [Mathematical Background](#mathematical-background)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

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
