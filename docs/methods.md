# Methods Reference

## 1. Grassberger-Procaccia Correlation Integral

The correlation dimension $D_2$ is estimated from:

$$C(r) = \lim_{N\to\infty} \frac{2}{N(N-1)} \sum_{i<j} \Theta(r - \|\mathbf{x}_i - \mathbf{x}_j\|)$$

**Implementation:** `scipy.spatial.cKDTree` for $O(N \log N)$ distance queries.  
**Scaling region:** Detected via Contiguous Gradient Max-Slope algorithm (minimum 10 points).  
**Slope estimator:** Theil-Sen robust regression on $\log C(r)$ vs $\log r$.

Reference: Grassberger & Procaccia (1983), doi:10.1103/PhysRevLett.50.346

---

## 2. Magnitude of Completeness

MAXC method: $M_c$ = bin with maximum count in the frequency-magnitude distribution.

Reference: Wiemer & Wyss (2000), doi:10.1785/0119990114

---

## 3. b-value Estimation

Aki (1965) maximum-likelihood estimator:

$$b = \frac{\log_{10} e}{\langle m \rangle - (M_c - \Delta m / 2)}, \quad \sigma_b = \frac{b}{\sqrt{N}}$$

Reference: Aki (1965), doi:10.1785/BSSA0550000423

---

## 4. Coordinate Transformation

WGS84 (lat/lon/depth) → local metric (km):

- $k_{\rm lat} = 111.1$ km/degree
- $k_{\rm lon} = 111.1 \cos(\bar{\phi} \cdot \pi/180)$ km/degree
- Normalization to unit cube $[0,1]^3$ for isotropy

---

## 5. Bayesian Inference of $D_3$

**Likelihood:** GP-scaling residuals in log-log space
$$\log \mathcal{L}(\theta | D_2^{\rm obs}) \propto -\frac{1}{2\hat{s}^2}\sum_k\left[\log C(r_k) - D_3 \log r_k - \hat{a}\right]^2$$

where $\hat{a}$ is the log-amplitude nuisance parameter (marginalized analytically)
and $\hat{s}^2 = \mathrm{Var(residuals)} + 10^{-4}$ (adaptive variance with safety floor).

**Prior:** Uniform $[1.5, 3.0]$

**Sampler:** `emcee` ensemble MCMC, 32 walkers, 10,000 steps, 2,000 burn-in

**Convergence:** $\hat{R} < 1.01$, ESS $> 5{,}000$, KL divergence from prior

---

## 6. Fisher Information Barrier

The critical precision threshold:

$$\sigma_c = 2.3 \pm 0.4 \text{ km}$$

Defined as the inflection point of $P_{\rm bnd}(\sigma_h)$ — the fraction of posterior
mass concentrated at the Euclidean bound $D_3 = 3.0$.

**Empirical validation:** SCSN California catalog, 9 degradation levels $\sigma_h \in [0.1, 10.0]$ km.

---

## 7. Bootstrap Uncertainty

Block bootstrap with replacement, $B = 1{,}000$ resamples.  
Standard error: $\text{SEM} = \sigma(D_2^{(b)}) / \sqrt{B}$  
95% CI: $D_2 \pm 1.96 \cdot \text{SEM}$
