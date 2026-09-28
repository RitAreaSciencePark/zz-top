# Vectorizations — Mathematical Reference

This document provides the mathematical definitions for all persistence
diagram vectorizers in `zztop.vectorizations`.

---

## 1. Standard Vectorizers

### 1.1 Persistence Image  (`PersistenceImage`)

*Reference: Adams et al., "Persistence Images: A Stable Vector Representation
of Persistent Homology", JMLR 2017.*

Given a persistence diagram $D_d = \{(b_i, d_i)\}$ in dimension $d$, define
the **birth–persistence** transform $T(b_i, d_i) = (b_i,\; p_i)$ where
$p_i = d_i - b_i$.

A **weighting function** $w(b, p)$ assigns importance to each point.  The
default is the linear ramp $w(b, p) = p$.

The persistence image is:

$$
\rho(x, y) = \sum_i w(b_i, p_i) \cdot \phi_\sigma(x - b_i,\; y - p_i)
$$

where $\phi_\sigma$ is a 2-D Gaussian with standard deviation $\sigma$.  This
continuous function is evaluated on a regular $(n_x \times n_y)$ pixel grid.

**Parameters:** `resolution=(ny, nx)`, `sigma`, `weight`, `birth_range`,
`pers_range`.

---

### 1.2 Persistence Landscape  (`PersistenceLandscape`)

*Reference: Bubenik, "Statistical Topological Data Analysis using Persistence
Landscapes", JMLR 2015.*

For each bar $(b_i, d_i)$, define the **tent function**:

$$
\Lambda_i(t) = \max\!\big(0,\; \min(t - b_i,\; d_i - t)\big)
$$

The $k$-th **landscape function** is the $k$-th largest value at each $t$:

$$
\lambda_k(t) = \text{kmax}_{i}\; \Lambda_i(t)
$$

The vectorizer evaluates $\lambda_1, \ldots, \lambda_K$ on a grid of
$R$ points in $[t_{\min}, t_{\max}]$.

**Output size:** $K \times R$ per dimension.

---

### 1.3 Silhouette  (`Silhouette`)

*Reference: Chazal et al., "Stochastic Convergence of Persistence Landscapes
and Silhouettes", SoCG 2014.*

$$
\phi^{(p)}(t) = \frac{\sum_i (d_i - b_i)^p \cdot \Lambda_i(t)}
                      {\sum_i (d_i - b_i)^p}
$$

where $\Lambda_i$ is the tent function.  The parameter $p$ controls
emphasis on long-lived features ($p > 1$) vs. equal weighting ($p = 1$).

**Output size:** $R$ per dimension.

---

### 1.4 Persistence Entropy  (`PersistenceEntropy`)

Shannon entropy of normalised bar lengths:

$$
H_d = -\sum_i p_i \log p_i, \qquad
p_i = \frac{\ell_i}{\sum_j \ell_j}, \qquad
\ell_i = d_i - b_i
$$

With `normalize=True`, the output is divided by $\log N_d$ (number of bars)
to normalise to $[0, 1]$.

**Output size:** 1 scalar per dimension.

---

### 1.5 Amplitude  (`Amplitude`)

A scalar summary measuring the "size" of a diagram.

| Metric | Formula |
|--------|---------|
| `wasserstein` | $\displaystyle A_p = \Big(\sum_i \lvert d_i - b_i\rvert^p\Big)^{1/p}$ |
| `bottleneck` | $\displaystyle A_\infty = \max_i \lvert d_i - b_i\rvert$ |
| `landscape` | $\displaystyle \frac{1}{2}\max_i \lvert d_i - b_i\rvert$ |

**Output size:** 1 scalar per dimension.

---

### 1.6 Persistence Statistics  (`PersistenceStatistics`)

Basic statistics of the birth, death, and persistence distributions:

$$
\text{vec}_d = \big[N_d,\; \bar{b},\; \sigma_b,\; \bar{d},\; \sigma_d,\;
\bar{p},\; \sigma_p,\; \max(p),\; H_d\big]
$$

**Output size:** 9 features per dimension.

---

## 2. Curve-Based Vectorizers

All curves are sampled on a regular grid $t_1, \ldots, t_R$ in
$[t_{\min}, t_{\max}]$.

### 2.1 Betti Curve  (`BettiCurve`)

$$
\beta_d(t) = \#\{i : b_i \le t < d_i\}
$$

### 2.2 Life Curve  (`LifeCurve`)

$$
L_d(t) = \sum_{i:\, b_i \le t < d_i} (d_i - b_i)
$$

### 2.3 Birth Curve  (`BirthCurve`)

$$
B_d(t) = \#\{i : b_i \le t\}
$$

### 2.4 Death Curve  (`DeathCurve`)

$$
D_d(t) = \#\{i : d_i \le t\}
$$

### 2.5 Midlife Curve  (`MidlifeCurve`)

Kernel density estimate of midpoints $m_i = (b_i + d_i) / 2$:

$$
M_d(t) = \frac{1}{h\sqrt{2\pi}} \sum_i
\exp\!\left(-\frac{(t - m_i)^2}{2h^2}\right)
$$

### 2.6 Multiplicative Life Curve  (`MultiplicativeLifeCurve`)

$$
\mathcal{M}_d(t) = \sum_{i:\, b_i \le t < d_i} \log(d_i - b_i)
$$

(Log-product of alive-bar lifetimes.)

### 2.7 Persistence Curve  (`PersistenceCurve`)

Generic framework: user supplies a function
$f(\mathbf{b}, \mathbf{d}, \mathbf{p}, t)$ evaluated on the alive bars at
each grid point.

---

## 3. Zigzag-Specific Descriptors

These exploit the **multi-layer (frame) structure** of zigzag persistence.
Birth/death values are frame indices (integers).

### 3.1 Betti Profile  (`BettiProfile`)

$$
\beta_d(t) = \#\{i : b_i \le t < d_i\}, \qquad t = 0, 1, \ldots, T-1
$$

Same formula as the Betti curve but evaluated on integer frame indices.

### 3.2 Birth Frequency  (`BirthFrequency`)

*Inspired by Eq. 5-6 of Gardinazzi et al. (2024).*

$$
f_d(t) = \frac{\#\{i : b_i = t\}}{N_d}
$$

Fraction of features born at each frame.

### 3.3 Persistence Profile  (`PersistenceProfile`)

*Inspired by the inter-layer persistence concept (Eq. 7-8 of Gardinazzi
et al. 2024), simplified from 2-D matrix to 1-D profile.*

$$
P_d(t) = \frac{1}{\beta_d(t)} \sum_{i:\, b_i \le t < d_i} (d_i - b_i)
$$

Mean persistence of bars alive at frame $t$.

### 3.4 Turnover Rate  (`TurnoverRate`)

$$
T_d(t) = \frac{\text{births}_d(t) + \text{deaths}_d(t)}
              {\beta_d(t-1) + \beta_d(t) + 1}
$$

Measures topological activity (features appearing and disappearing) at each
frame.  High turnover = rapid reshaping.

### 3.5 Effective Persistence Image  (`EffectivePersistenceImage`)

*Inspired by Eq. 4 of Gardinazzi et al. (2024).*

Standard persistence image with weights modulated by birth frequency:

$$
w_i = (d_i - b_i) \cdot f_d(b_i)
$$

where $f_d(b_i)$ is the birth-frequency at frame $b_i$.

### 3.6 Cumulative Persistence  (`CumulativePersistence`)

$$
C_d(t) = \sum_{i:\, d_i \le t} (d_i - b_i)
$$

Running total of lifetime for features that have died by frame $t$.

---

## 4. Experimental Descriptors

These are **not** in `__all__` and should be imported explicitly from
`zztop.vectorizations.zigzag`.

### 4.1 Layer Transition Entropy  (`LayerTransitionEntropy`)

At each frame $t$, form the event vector
$\mathbf{c} = [n_{\text{births}}, n_{\text{deaths}}, n_{\text{alive}}]$
and compute its Shannon entropy:

$$
E(t) = -\sum_k \frac{c_k}{\lVert\mathbf{c}\rVert_1}
\log \frac{c_k}{\lVert\mathbf{c}\rVert_1}
$$

### 4.2 Persistence Flux  (`PersistenceFlux`)

$$
\Phi_d(t) = P_{\text{total},d}(t) - P_{\text{total},d}(t-1)
$$

where $P_{\text{total},d}(t) = \sum_{i:\, b_i \le t < d_i}(d_i - b_i)$.

Positive flux = growing persistence; negative = shrinking.

### 4.3 Cross-Dimension Correlation  (`CrossDimCorrelation`)

Pearson $r$ between Betti profiles of different dimensions:

$$
r_{d_1, d_2} = \operatorname{corr}\big(\beta_{d_1}(0:T),\;
\beta_{d_2}(0:T)\big)
$$

Output: one scalar per dimension pair.

### 4.4 Lifetime Spectrum  (`LifetimeSpectrum`)

Histogram of bar lifetimes $\{d_i - b_i\}$ with $B$ bins.

---

## 5. Optional Torch Support

All vectorizers have a `.to_torch(X)` method that returns a
`torch.Tensor` (requires PyTorch installed).

For more control, use:

```python
from zztop.vectorizations._torch import TorchVectorizer, differentiable_transform

# Wrap any vectorizer
tpi = TorchVectorizer(PersistenceImage(...), device="cuda")
features = tpi.fit_transform(diagrams)

# Or convert post-hoc
tensor = differentiable_transform(fitted_vectorizer, X, requires_grad=True)
```

---

## References

- H. Adams et al., *Persistence Images: A Stable Vector Representation of
  Persistent Homology*, JMLR 18(8):1–35, 2017.
- P. Bubenik, *Statistical Topological Data Analysis using Persistence
  Landscapes*, JMLR 16(3):77–102, 2015.
- F. Chazal et al., *Stochastic Convergence of Persistence Landscapes and
  Silhouettes*, SoCG 2014.
- M. Biagetti et al., *The Persistence of Large Scale Structures I:
  Primordial non-Gaussianity*, JCAP 04 (2021) 061, arXiv:2009.04819.
- C. Gardinazzi et al., *Topological Analysis for Detecting Anomalies in
  Text and LLMs*, ICML 2024 TDA & Beyond Workshop, arXiv:2410.11042.
