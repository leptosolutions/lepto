<div align="center">
  <img src="lepto/assets/logo.png" alt="Lepto Logo" width="100"/>
</div>

# Lepto

Lepto is a white-box modeling toolkit for interpretable Generalized Linear Models (GLMs), supporting both standard and behaviour GLMs. It is designed for transparency, flexibility, and ease of use, with both a graphical user interface (GUI) and a Python API.
Lepto was orginally developed for actuarial analysis but can be used in other fields.

---

## Features

- **Standard GLM**: Gaussian, Poisson, Gamma, Tweedie, Binomial families
- **Behaviour GLM**: For models with behaviour variables (e.g., price sensitivity)
- **Structured Regularization**: Penalty matrices for continuous, categorical, and graph variables
- **Monotonicity Constraints**: Enforce monotonic relationships (e.g., price/probability)
- **Partial Coefficient Freezing**: Offset support for fixed coefficients
- **Sample Weights**: Weighted fitting for all models
- **Scikit-learn Compatible**: Follows estimator API for easy integration
- **Interactive GUI**: Data import, model setup, review, and export
- **Export**: Model summaries and results to Excel or DataFrame

---

## Installation

Lepto requires Python 3.8+ and can be installed via pip:

```sh
pip install lepto
```

For best performance, set environment variables to control thread usage:

```sh
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

---

## Quick Start

### Launching the GUI

**From the terminal:**
```sh
lepto-app
```

### Using the API

```python
import pandas as pd
import numpy as np
from lepto.standard.model.linear_model import GLMDiff

X = pd.DataFrame({'age': [25, 40, 60], 'region': ['A', 'B', 'A']})
y = np.random.poisson(lam=2.0, size=3)
penalty_choice = {'age': {'penalty': 'continuous'}}
model = GLMDiff(family="poisson", lam=0.01, penalty_choice=penalty_choice, nbins=2)
model.fit(X, y)
preds = model.predict(X)
print(model.summary)
```

---

## GUI Overview

The Lepto GUI allows you to:
- Import and preprocess data
- Set up and configure models
- Review model results and diagnostics
- Export model summaries to Excel or DataFrame

<!-- ![Lepto GUI Screenshot](docs/lepto_gui_screenshot.png) -->

---

## API Reference

- `lepto.standard.model.linear_model.GLMDiff`: Standard GLM estimator
- `lepto.standard.model.linear_model.transform_json_to_df`: Export model summary to DataFrame
- `lepto.gui.Lepto_GUI.run()`: Launch the GUI from Python

See the [docs/](../docs/) folder for more details and advanced usage.

---

## Model Mathematics

### Standard GLMs

Standard GLM is a **Generalized Linear Model (GLM)** with an **optional structured quadratic penalty**, designed to be compatible with the **scikit-learn estimator API** (`BaseEstimator`, `RegressorMixin`).

It supports:
- Classical GLMs (Gaussian, Poisson, Gamma, Tweedie, Binomial)
- Observation/sample weights
- Structured regularization via a penalty matrix
- Partial freezing of coefficients via offsets

The predictions are computed via:

$\eta = g(X\beta)$

Given a design matrix $X \in \mathbb{R}^{n \times p}$ and a response vector $y \in \mathbb{R}^{n}$, the model estimates coefficients $\beta \in \mathbb{R}^{p}$ by minimizing a penalized negative log-likelihood:

$$
\hat{\beta}=\arg\min_{\beta}\left[-\sum_{i=1}^{n} \ell(y_i, x_i^\top \beta)+\lambda \lVert D\beta \rVert_2^2\right]
$$

Where:
- $D$ is a structured penalty matrix based on adjacency matrix for each variable in the model
- $\lambda$ controls the strength of regularization

This corresponds to **penalized maximum likelihood estimation**.

$D$ is built via an adjacency matrix for each variable, the correspondant penalization is:
- Continuous variables:  $\lambda \sum\limits_{i}^{ } (\beta_{i+1} - \beta_i)^2$
- Categorical variables:  $\lambda \sum\limits_{i>k}^{ } (\beta_{i} - \beta_k)^2$
- Graph variables:  $\lambda \sum\limits_{(i, k) \in G}^{ } (\beta_{i} - \beta_k)^2$

### Behaviour GLMs

Behaviour GLM is two combined **Generalized Linear Models (GLM)** with **optional structured quadratic penalty**, designed to be compatible with the **scikit-learn estimator API** (`BaseEstimator`, `RegressorMixin`).
It is designed for cases with a behaviour variable (e.g., price, price evolution). The behaviour variable interacts with all other variables, resulting in two model parts:
- A non-behaviour part describing the problem without the behaviour variable
- A behaviour part describing the behaviour aspect, with the behaviour variable interacting with all selected variables

A monotonicity constraint can be set to ensure, for example, that probability always decreases with price.

The predictions are computed via:

$\eta = sigmoid(X\beta_1 + q X \beta_2 )$

Where:
- $X\beta_1$ is the non-behaviour part
- $X \beta_2$ is the behaviour part
- $q$ is the behaviour variable

$$(\hat{\beta}_1,\hat{\beta}_2)=\arg\min_{\beta_1,\beta_2}\Big\{-\sum_{i=1}^n w_i\Big[y_i\log\sigma(z_i)+(1-y_i)\log(1-\sigma(z_i))\Big]+\lambda\Big(\|D_1\beta_1+d_1\|_2^2+\|D_2\beta_2+d_2\|_2^2\Big)\Big\} + \lambda_{behaviour} \sum_{i=1}^n (h_i^2)$$

Where:
- $z_i = x_{1i}^\top\beta_1 + q_i\,(x_{2i}^\top\beta_2),\qquad\sigma(z)=\frac{1}{1+e^{-z}}$
- $h_i = softplus(-k*g_i)/k$
- $g_i = x_{2i}^\top \beta_2 $

---

## Help and Support

For questions, bug reports, or feature requests, please open an issue on the [GitHub repository](https://github.com/lepto_solutions/lepto) or contact the maintainer.

---

## Contributing

Contributions are welcome! Please submit pull requests, report issues, or suggest features via GitHub.

---

## License

Lepto is released under the MIT License. See [LICENSE](../LICENSE) for details.
