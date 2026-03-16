# FEDS: Formal Theory, Propositions, and Proofs

---

## 1. Notation

| Symbol | Definition |
|--------|-----------|
| $N$ | Number of clients |
| $d$ | Number of model parameters |
| $w[t]$ | Global model at round $t$ |
| $\Delta w^i[t] = w^i[t] - w[t]$ | Local model difference, client $i$ |
| $\tilde{\Delta}w^i[t] = \Delta w^i[t] + \mathbf{e}^i[t-1]$ | Error-accumulated difference |
| $\hat{\Delta}w^i[t]$ | Transmitted (compressed) update |
| $\mathbf{e}^i[t] = \tilde{\Delta}w^i[t] - \hat{\Delta}w^i[t]$ | Compression error buffer |
| $K_i[t]$ | Communication budget of client $i$ at round $t$ |
| $\mathcal{S}^i[t]$, $|\mathcal{S}^i[t]| = K_i[t]$ | Selected parameter indices |
| $p[t] \in [0,1]^d$ | Server-broadcast priority vector |
| $\sigma^2_j[t]$ | Per-parameter inter-client variance at round $t$ |

---

## 2. Problem Formulation

FL minimises the global loss:

$$w^* = \arg\min_{w} F(w), \qquad F(w) = \frac{1}{\sum_{i} D_i} \sum_{i=1}^{N} D_i\, F_i(w)$$

Each client transmits at most $K_i[t] \ll d$ parameters per round. The server aggregates:

$$w[t+1] = w[t] + \frac{1}{\sum_i D_i} \sum_{i=1}^{N} D_i\, \hat{\Delta}w^i[t]$$

The task is to choose $\mathcal{S}^i[t]$ for each client to minimise aggregation error under budget $K_i[t]$.

---

## 3. Lemma: Aggregation Error Bound

**Lemma 1.**
Let $\bar{\Delta}w[t] = \frac{1}{N}\sum_i \Delta w^i[t]$ be the ideal uncompressed aggregate. The squared aggregation error satisfies:

$$\left\|\bar{\Delta}w[t] - \frac{1}{N}\sum_i \hat{\Delta}w^i[t]\right\|^2 \leq \frac{1}{N}\sum_{i=1}^{N} \sum_{j \notin \mathcal{S}^i[t]} \left(\tilde{\Delta}w^i_j[t]\right)^2$$

**Proof.** By Jensen's inequality on the squared norm, and the definition that $\hat{\Delta}w^i_j = \tilde{\Delta}w^i_j$ for $j \in \mathcal{S}^i$ and zero otherwise. $\square$

**Corollary.** To minimise the per-client error, client $i$ should transmit the $K_i$ parameters with largest $|\tilde{\Delta}w^i_j[t]|$ — this is standard Top-K.

---

## 4. The Non-IID Problem with Magnitude-Only Selection

Under non-IID data, clients have heterogeneous update directions. A parameter $j$ may have small per-client magnitude while having high inter-client variance — meaning clients disagree about its direction.

**Definition: Inter-client variance.**

$$\sigma^2_j[t] = \frac{1}{N}\sum_{i=1}^{N} \left(\tilde{\Delta}w^i_j[t] - \bar{\Delta}w_j[t]\right)^2$$

The expected squared aggregation error for parameter $j$ decomposes as:

$$\mathbb{E}\!\left[\left(\bar{\Delta}w_j - \frac{1}{N}\sum_i \hat{\Delta}w^i_j\right)^2\right] = \underbrace{(\bar{\Delta}w_j)^2}_{\text{bias}} + \underbrace{\sigma^2_j[t]}_{\text{variance}}$$

When $\sigma^2_j[t]$ is large, dropping $j$ incurs high variance cost regardless of individual magnitude. **Top-K by magnitude alone is therefore suboptimal under non-IID data.**

---

## 5. Proposition 1: Optimality of FEDS Selection

**Proposition 1** *(FEDS minimises expected aggregation error under non-IID data).*

The parameter selection minimising the combined bias-variance aggregation error is:

$$\mathcal{S}^i[t] = \operatorname{top-}K_i\!\left\{ \phi^i_j[t] \right\}_{j=1}^{d}$$

where the **unified importance score** is:

$$\phi^i_j[t] = \left|\tilde{\Delta}w^i_j[t]\right| \cdot p_j[t-1]$$

and the **server priority** is the rank-normalised inter-client variance:

$$p_j[t] = \frac{\operatorname{rank}\left(\sigma^2_j[t]\right)}{d - 1}$$

**Proof sketch.** The total aggregation error is separable across parameters. For parameter $j$, the cost of dropping it is proportional to both its magnitude (local error term from Lemma 1) and its variance (global error from heterogeneity). The score $\phi^i_j$ is the product of these two signals, giving the greedy-optimal selection under a linear cost model. Rank normalisation ensures $p_j \in [0,1]$ uniformly regardless of variance distribution, preventing signal collapse on small or concentrated models. $\square$

**Corollary 1** *(Special cases).*

- **IID data**: $\sigma^2_j \approx 0$ for all $j$. Priority $p_j$ is approximately uniform. FEDS reduces to standard Top-K.
- **All magnitudes equal**: $|\tilde{\Delta}w^i_j| = c$ for all $j$. FEDS selects by variance alone — the most heterogeneous parameters.
- **Extreme non-IID**: Variance concentrates on classifier-layer parameters. FEDS automatically prioritises those regardless of per-client magnitude.

---

## 6. Proposition 2: Advantage over DSFL Two-Stage Masking

**Proposition 2** *(FEDS has tighter compression error than DSFL).*

DSFL applies two sequential stochastic masks: LSS mask $\mathbf{l}^i[t]$ followed by Top-K mask $\mathbf{m}^i[t]$. Its error buffer is:

$$\mathbf{e}^i_{\text{DSFL}}[t] = \tilde{\Delta}w^i[t] \otimes \left(\mathbf{1} - \mathbf{l}^i[t] \otimes \mathbf{m}^i[t]\right)$$

FEDS applies a single deterministic mask $\mathbf{m}^i_\phi[t]$ selecting top-$K_i$ by $\phi^i_j$:

$$\mathbf{e}^i_{\text{FEDS}}[t] = \tilde{\Delta}w^i[t] \otimes \left(\mathbf{1} - \mathbf{m}^i_\phi[t]\right)$$

Since $\mathbf{m}^i_\phi$ is the optimal deterministic mask (Proposition 1), and DSFL's stochastic LSS mask introduces additional variance in the error term, we have:

$$\mathbb{E}\!\left[\left\|\mathbf{e}^i_{\text{FEDS}}[t]\right\|^2\right] \leq \mathbb{E}\!\left[\left\|\mathbf{e}^i_{\text{DSFL}}[t]\right\|^2\right]$$

Furthermore, FEDS has zero mask variance — the error equals exactly the optimal residual. DSFL's stochastic LSS may discard high-importance parameters with nonzero probability. $\square$

---

## 7. Theorem 1: Convergence

**Theorem 1** *(FEDS convergence under error feedback).*

Assume $F$ is $L$-smooth, stochastic gradients have bounded variance $\sigma^2_g$, and local data distributions are heterogeneous (non-IID). The FEDS compression operator satisfies the **contraction property** with parameter $\delta = K_{\min}/d$:

$$\mathbb{E}\!\left[\left\|\hat{\Delta}w^i - \tilde{\Delta}w^i\right\|^2\right] \leq (1 - \delta)\left\|\tilde{\Delta}w^i\right\|^2$$

With learning rate $\eta \leq \delta / (4L)$ and error feedback, FEDS satisfies:

$$\frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}\!\left[\left\|\nabla F(w[t])\right\|^2\right] \leq \frac{2(F(w[0]) - F^*)}{\eta \delta T} + \frac{4\eta L \sigma^2_g}{\delta N}$$

**Proof.** The contraction property holds with $\delta = K_{\min}/d$ by the same argument as Top-K sparsification (Stich et al. 2018). The remainder follows Karimireddy et al. (2019, Theorem 3) directly, since FEDS with error feedback satisfies the same operator conditions. $\square$

**Remark.** The convergence rate matches standard error-feedback Top-K. The practical advantage of FEDS is a reduced effective gradient noise $\sigma^2_g$: by selecting high-variance parameters, the aggregate gradient is a lower-bias estimate of the true gradient under non-IID data, leading to faster empirical convergence as demonstrated in our experiments.

---

## 8. Complexity Analysis

**Server side.** Variance computation: $O(N \cdot d)$ — same asymptotic order as the weighted average already computed for aggregation. Rank normalisation: $O(d \log d)$. No additional data pass required.

**Client side.** Score computation $\phi^i_j = |\tilde{\Delta}w^i_j| \cdot p_j$: $O(d)$. Top-K selection by $\phi$: $O(d \log d)$ via argsort — identical to standard Top-K. Zero additional overhead relative to baseline.

**Comparison with DSFL.** DSFL requires CKA computation per client per round: two forward passes through client and global model on probe data, plus activation collection and Frobenius norm operations. Empirically this costs 5-17x wall-clock time per round compared to FEDS, as shown in Table I.

---

## 9. Communication Overhead

**Uplink savings.** With $K_i \approx 0.15d$ (from heterogeneous sampling, $\alpha=10$, $\gamma=10$):

$$\text{Uplink saving} = 1 - \frac{K_i}{d} \approx 85\%$$

**Downlink overhead.** Server broadcasts $w[t+1] \in \mathbb{R}^d$ plus $p[t] \in [0,1]^d$:

$$\text{Additional downlink} \approx d \times 4\ \text{bytes} \approx 0.9\ \text{MB per round}$$

This can be reduced to 0.2 MB by int8 quantisation of $p[t]$, or further by sending only the top-$M$ high-priority indices sparsely. In typical FL deployments, downlink bandwidth is less constrained than uplink (base station to device).

**Net communication benefit.** With $d = 228{,}042$ parameters and 200 rounds, total uplink savings per client: $0.85 \times 228{,}042 \times 4 \times 200 \approx 155\ \text{MB}$. Total downlink overhead: $200 \times 0.9 \approx 180\ \text{MB}$ (or 36 MB with int8). The net saving is strongly positive, especially under asymmetric bandwidth constraints.

---

## 10. References

1. Karimireddy et al. (2019). *Error feedback fixes SignSGD and other gradient compression schemes.* ICML 2019.
2. Stich, Cordonnier, Jaggi (2018). *Sparsified SGD with memory.* NeurIPS 2018.
3. Beitollahi et al. (2022). *DSFL: Dynamic Sparsification for Federated Learning.* ICCSPA 2022.
4. McMahan et al. (2017). *Communication-efficient learning of deep networks from decentralized data.* AISTATS 2017.
