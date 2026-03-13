# RESEARCH PROPOSAL

## Loss-Feedback Adaptive Sparsification for Communication-Efficient Federated Learning over Heterogeneous Wireless Networks

**Target Venue:** IEEE GLOBECOM 2026  
**Deadline:** April 1, 2026  
**Timeline:** 19 Days

**Md Akmol Masud**  
Incoming M.A.Sc. Student, Department of Electrical and Computer Engineering  
Queen's University, Kingston, Ontario, Canada  

*Supervisor: Prof. Ning Lu (Canada Research Chair, Tier 2, Future Communication Networks)*

---

## 1. Executive Summary

Federated learning (FL) systems deployed over bandwidth-constrained wireless networks face a fundamental tension: clients must transmit model updates frequently, but their communication capacity varies dynamically across rounds. DSFL (Dynamic Sparsification for FL, Beitollahi et al., ICCSPA 2022) partially addresses this by introducing a two-level sparsification pipeline — Layer-wise Similarity Sparsification (LSS) using CKA and per-client top-K — but assigns each client's sparsification rate K from a fixed, pre-specified statistical distribution. This static assumption is the core limitation we attack.

We propose **ADAFL** (Adaptive Sparsification for Federated Learning): a loss-feedback driven mechanism that adjusts each client's K dynamically using only the training signal already present in DSFL's pipeline, requiring zero additional communication overhead. When a client's local loss improves significantly, the algorithm allows it to compress more aggressively in the next round; when loss stagnates, compression is relaxed to send richer updates. This closes the gap DSFL explicitly leaves as future work, is implementable directly from Mahdi Beitollahi's publicly available codebase, and fits naturally within Prof. Lu's research agenda bridging DSFL and the bandit-based scheduling work of Juaren Steiger.

---

## 2. Research Context and Lab Lineage

### 2.1 The Queen's University FL Thread

This proposal sits at the direct intersection of two completed research threads in Prof. Lu's lab. Understanding this lineage is essential to appreciating why the contribution is timely, novel, and well-scoped.

| Paper | Venue | Contribution | Limitation Left Open |
|-------|-------|--------------|----------------------|
| DSFL | ICCSPA 2022 (Beitollahi) | Two-level sparsification: LSS (CKA) + per-client top-K + error accumulation | K assigned from fixed truncated normal — no learning |
| FLAC | GLOBECOM 2022 (Beitollahi) | Autoencoder compression with convergence guarantee | Different mechanism; no adaptation of K |
| POSS | INFOCOM 2023 (Steiger) | Constrained bandit framework with switching costs | No FL compression component |
| Backlogged Bandits | INFOCOM 2024 (Steiger) | Queueing + bandit formulation for network scheduling | No FL compression component |
| **This work (ADAFL)** | GLOBECOM 2026 (Masud) | Loss-feedback adaptive K within DSFL's LSS+top-K pipeline | MAB extension (Phase 2, INFOCOM 2027) |

### 2.2 The Publication Gap and Opportunity

Mahdi Beitollahi left Queen's in late 2022 for Huawei Noah's Ark Lab Montreal, where his work pivoted entirely to foundation models and federated unlearning. Juaren Steiger completed his PhD in 2024–2025 and is now a postdoctoral researcher at Penn State. Prof. Lu's lab has a 2+ year gap in FL communication-efficiency publications. ADAFL directly restarts this thread — arriving at Queen's with a submission already in review demonstrates precisely the kind of proactive research readiness that accelerates a master's program into a PhD trajectory.

---

## 3. Problem Statement

### 3.1 DSFL's Core Limitation

In DSFL, each client *i*'s sparsification rate at round *t* is modeled as:

**K<sub>i</sub>[*t*] ~ Truncated-Normal(μ = *d*/α, σ = *d*/γ, *a* = 0, *b* = *d*)**

where *d* is the model dimension and α, γ are fixed scalars. This distribution is pre-specified before training begins and never updates. The key failure modes are: (1) it cannot respond to changes in channel conditions across rounds; (2) it ignores the training signal entirely — a client making excellent progress uses the same compression policy as a client whose training has stalled; and (3) the distribution parameters α and γ require manual tuning per deployment scenario.

DSFL's authors explicitly acknowledge this in the conclusion: *"it is still unknown how we can fully exploit this correlation for the improvement of FL"* and flag the static nature of K as a limitation left for future work. Our contribution directly answers this call.

---

## 4. Proposed Method: ADAFL

### 4.1 Core Idea

ADAFL replaces DSFL's static K-sampling with a feedback-driven update rule. The intuition is simple: if a client's local loss improved substantially in round *t*, its current update was informative — it can afford to compress more next round. If loss barely changed, the update was weak — send richer data by relaxing compression.

**K<sub>i</sub>[*t*+1] = clip( K<sub>i</sub>[*t*] - η · (ΔL<sub>i</sub>[*t*] − τ), K<sub>min</sub>, K<sub>max</sub> )**

where **ΔL<sub>i</sub>[*t*] = L<sub>i</sub>[*t*−1] − L<sub>i</sub>[*t*]** is the local loss improvement of client *i* in round *t*, τ is a threshold (target improvement rate), η is a learning rate for the K-update, and K<sub>min</sub>, K<sub>max</sub> define the feasible compression range. This update runs entirely on the client using locally available information — no additional communication to the server is required. Note the negative sign: larger loss improvements lead to *greater* compression (smaller K).

### 4.2 Integration with DSFL's Pipeline

ADAFL plugs into DSFL's existing Algorithm 1 with a single modification — replacing line 14 (K<sub>i</sub> = C<sub>i</sub>[*t*]) with the adaptive rule above. The rest of the pipeline — LSS masking, top-K sparsification, error accumulation, and weighted aggregation — remains unchanged. This is a minimal surgical change that preserves all of DSFL's theoretical and engineering properties while adding the adaptive layer.

| DSFL Step | DSFL Original | ADAFL Modification |
|-----------|---------------|---------------------|
| Line 7 | Download global model *w*[*t*] | Same — unchanged |
| Line 8 | Perform E local updates | Same — additionally record L<sub>i</sub>[*t*] |
| Line 9 | Compute ΔW<sub>i</sub>[*t*] | Same — unchanged |
| Line 11 | Compute LSS mask *l*<sub>i</sub>[*t*] via CKA | Same — unchanged |
| **Line 14** | K<sub>i</sub> = C<sub>i</sub>[*t*] (from truncated normal) | **K<sub>i</sub>[*t*+1] = clip(K<sub>i</sub>[*t*] - η·(ΔL<sub>i</sub>[*t*]−τ), K<sub>min</sub>, K<sub>max</sub>) ← CHANGE** |
| Lines 15–18 | Top-K sparsification, upload, error update | Same — K<sub>i</sub>[*t*] used from updated rule |

### 4.3 Why This Works: Intuition

Consider two clients in round *t*. **Client A** (high heterogeneity, poor channel) achieves ΔL<sub>A</sub> = 0.01, well below threshold τ = 0.05. ADAFL *increases* K<sub>A</sub> (since ΔL - τ is negative) — client A sends **denser** updates (more parameters) next round to catch up. **Client B** (well-aligned data, good channel) achieves ΔL<sub>B</sub> = 0.12, above threshold. ADAFL *decreases* K<sub>B</sub> — client B is learning well and can afford to send **sparser** (fewer parameters) updates without hurting convergence. Over many rounds, each client's K settles near the level appropriate for its current condition without any manual tuning.

---

## 5. Related Work and Differentiation

The FL communication efficiency space is crowded. The following table positions ADAFL precisely against the closest existing work, all of which we identified through systematic literature search.

| Prior Work | Venue | What It Does | How ADAFL Differs |
|------------|-------|--------------|-------------------|
| DSFL (Beitollahi et al.) | ICCSPA 2022 | LSS + top-K sparsification; K from truncated normal | We make K adaptive via loss feedback — DSFL's own future work |
| Han et al. | ICDCS 2020 | Adaptive GS via online learning; non-IID aware | No MAB, no per-client K, no LSS layer-awareness |
| FedSparQ | 2025 | Adaptive threshold sparsification + error feedback + float16 | Layer-blind error feedback; no LSS; no loss-driven K |
| EFSkip | AAAI 2025 | Error feedback for arbitrary data heterogeneity | Error feedback only; no adaptive K; no LSS |
| AdapComFL | 2024 | Bandwidth prediction → adaptive compression | Requires bandwidth prediction module; we use only loss signal |
| FedIncSparse | 2025/2026 | Top-K sparse delta transmission + incremental updates | Delta-of-delta mechanism; no LSS; no loss feedback |
| DGCSFL | 2024 | K per client from pre/post training loss + client selection | Combines selection + K but discards LSS entirely; no DSFL lineage |
| FedLUAR | arXiv 2025 | Server-side layer recycling of sparse updates | Server-side intervention; different mechanism; no adaptive K |
| CS-UCB (Xia et al.) | IEEE TWC 2020 | MAB for client scheduling (who participates) | Scheduling ≠ compression rate; we select how much to compress |

The critical gap ADAFL fills: no existing work applies a training-signal-driven adaptive K update within DSFL's LSS + top-K pipeline. DGCSFL comes closest (loss-driven K) but abandons LSS and combines it with client selection, making it a different system. ADAFL is the natural, minimal extension of DSFL that its own authors called for.

---

## 6. Experimental Plan

**Benchmarking principle:** We use the **exact same experimental setting as DSFL** (Beitollahi et al., ICCSPA 2022) so that ADAFL and DSFL are compared fairly. This ensures reproducibility and allows reviewers to judge the gain from adaptive K alone, without confounding factors from different datasets, capacity models, or partitioning.

### 6.1 Datasets and Models (Same as DSFL)

- **MNIST** — CNN as in [19] (582,026 parameters); same as DSFL.
- **CIFAR-10** — CNN as in [1] (62,006 parameters); same as DSFL.
- **Speech Commands** — CNN as in [13] (27,303 parameters); same as DSFL.

All three datasets and architectures are taken directly from DSFL's paper and codebase (Flower framework). No additional datasets in the main evaluation; this keeps the benchmark strictly comparable.

### 6.2 Non-IID Partitioning (Same as DSFL)

- **10 clients** (same as DSFL).
- **Non-IID partition:** Labels distributed so that each pair of adjacent users has two unique labels that no other users have (DSFL's exact scheme). This matches the heterogeneity level reported in DSFL.
- No Dirichlet or other partitioning for the main results — we replicate DSFL's setup first. Optional: vary non-IID severity (e.g. Dirichlet α) in ablation if time permits.

### 6.3 Baselines

| Baseline | Why Included |
|----------|--------------|
| FedAvg (no compression) | Upper bound on accuracy — no communication constraint |
| Fixed top-K (uniform K for all clients) | Standard sparsification baseline |
| DSFL (truncated normal K) | Direct predecessor — primary comparison |
| Han et al. adaptive GS | Best non-bandit adaptive method; if code unavailable, compare to reported numbers |
| Random K assignment | Shows benefit of feedback vs. random policy |
| **ADAFL (ours)** | Proposed method |

### 6.4 Metrics

- Test accuracy vs. communication round
- Test accuracy vs. total bits transmitted (primary Globecom metric)
- Convergence round to reach target accuracy (e.g. 90% on MNIST)
- K trajectory per client over rounds (to visualize adaptive behavior)

### 6.5 Capacity / Communication Model (Same as DSFL)

We replicate DSFL's **exact** setup for fair benchmarking. In DSFL, each client *i*'s sparsification budget at round *t* is **K<sub>i</sub>[*t*] = C<sub>i</sub>[*t*]**, where C<sub>i</sub>[*t*] is sampled from:

**C<sub>i</sub>[*t*] ~ Truncated-Normal(μ = *d*/α, σ = *d*/γ, *a* = 0, *b* = *d*)**

(*d* = model dimension; α, γ = fixed scalars.) We use the **same** datasets, architectures, non-IID partition, learning rate, local epochs (E = 1 for MNIST/CIFAR-10, E = 5 for Speech), and **same α, γ and random seeds** when running DSFL so that DSFL's results are reproducible and match the paper. For **ADAFL**, we keep everything identical except: instead of K<sub>i</sub>[*t*] = C<sub>i</sub>[*t*], we use the loss-feedback rule with K<sub>i</sub>[*t*] ∈ [K<sub>min</sub>, K<sub>max</sub>] and K<sub>min</sub>, K<sub>max</sub> chosen to span the same range (e.g. [0, *d*] or consistent with the truncated normal). This isolates the effect of adaptive K. Optional (if time): vary α, γ to show robustness.

---

## 7. Theoretical Positioning

For a 6-page Globecom submission, a full convergence proof is not expected. The theoretical contribution is framed as follows:

**Claim:** Under the same assumptions as DSFL (bounded gradient dissimilarity, Lipschitz-smooth local objectives, bounded variance), ADAFL's adaptive K policy satisfies the sparsification conditions required for DSFL's convergence argument to hold. Specifically, the clipping operation in the K-update rule ensures K<sub>i</sub>[*t*] ∈ [K<sub>min</sub>, K<sub>max</sub>] at all times, preserving the bounded compression property DSFL's analysis requires. The loss-feedback term introduces a bias toward K values that historically correlated with progress, but does not violate the stochastic approximation conditions.

This is not a new theorem — it is a verification that ADAFL's modification does not break DSFL's existing guarantee. The full convergence analysis of ADAFL under dynamic K, including the effect of the feedback rule on the error accumulation term, is explicitly deferred to the journal extension (Phase 2). This is standard and accepted practice for Globecom papers.

The MAB extension (Phase 2) will then add formal regret bounds, borrowing from Steiger's POSS framework and Vaishnav's Budgeted UCB for the capacity-constrained setting. ADAFL's loss-feedback rule is designed to be a natural precursor to a UCB formulation: the ΔL<sub>i</sub>[*t*] signal that drives the K-update in ADAFL is the same signal that would serve as reward feedback in a bandit formulation.

---

## 8. Globecom Positioning and Scope Guardrails

**Conference fit.** ADAFL is intentionally scoped as a **Globecom-style, communications-centric extension** of DSFL. The paper keeps:

- **Exactly the same system model and experimental setup as DSFL** (datasets, architectures, non-IID partition, truncated-normal capacity model, Flower implementation).
- **Exactly the same sparsification pipeline** (LSS + top-K + error accumulation), changing only how the per-client sparsification rate K is chosen.

This framing makes it clear to reviewers that ADAFL is a **minimal, well-controlled modification** answering DSFL’s own future-work question about how to exploit the correlation between training signal and communication rate.

**Core novelty claim for Globecom.** The contribution should be stated narrowly and precisely:

- DSFL: **K<sub>i</sub>[t] is drawn once per round from a fixed truncated normal distribution**, independent of training signal and never updated.
- ADAFL: **K<sub>i</sub>[t] becomes a loss-feedback-driven state variable on each client**, updated as
  \(K_i[t+1] = \text{clip}\big(K_i[t] + \eta(\Delta L_i[t] - \tau), K_{\min}, K_{\max}\big)\)
  using **only local loss information and no extra communication**.

The headline should emphasize **better accuracy-per-bit and robustness under heterogeneous uplinks**, not “a new FL framework”.

**Experimental bar for acceptance.** Because the idea is incremental, the experiments must be:

- **Strictly comparable to DSFL**: same seeds, same capacity draws, same training hyperparameters, and same number of clients; only the K-selection rule changes.
- **Communication-centric**: plots of **test accuracy vs. global rounds** and **test accuracy vs. total transmitted bits** must show a **clear, consistent gap** between ADAFL and FedAvg (no compression), fixed top-K, DSFL (truncated-normal K), and optionally random-K.
- **Light but targeted ablations**: at most one small figure exploring sensitivity to \(\eta\), \(\tau\), or \([K_{\min}, K_{\max}]\), to avoid bloating the paper.

**Theory level appropriate for 6 pages.**

- The theoretical section should **only verify that ADAFL preserves DSFL’s convergence assumptions** (bounded compression via clipping, bounded gradient dissimilarity, etc.).
- Acknowledge explicitly that **full analysis of dynamic K and the feedback loop is deferred to the Phase-2 journal/INFOCOM extension**, where regret bounds and a bandit formulation will be developed.

**Scope guardrails vs. INFOCOM.**

- **Globecom (this paper):** practical, loss-feedback heuristic inside DSFL; focus on system design and communication-efficiency gains.
- **INFOCOM / journal (Phase 2):** formalize ADAFL as a **bandit-driven compression policy (MAB-DSFL)** on top of Steiger’s POSS framework, with regret and convergence analysis and potentially joint scheduling+compression.

This separation prevents the Globecom paper from becoming over-ambitious while clearly signaling a rich, theory-heavy follow-up line for reviewers.

---

## 9. 19-Day Execution Plan

| Days | Focus | Deliverable |
|------|-------|-------------|
| 1–2 | Lock system model and algorithm. Read DGCSFL + Han et al. Write related-work paragraph. Pin down τ and η defaults. Replicate DSFL's exact setup (datasets, capacity, non-IID). | Algorithm pseudocode (final) |
| 3–5 | Implement ADAFL in Mahdi's codebase. Replace K = C<sub>i</sub>[*t*] with loss-feedback rule. Use same capacity sampling as DSFL. Test on MNIST (same architecture and non-IID as DSFL). | Working code, first MNIST result |
| 6–8 | Run CIFAR-10 and Speech Commands with same DSFL settings. Full baseline suite on MNIST: DSFL, fixed top-K, ADAFL, FedAvg (no compression). | Main result figures (all three datasets) |
| 9–11 | Generate accuracy vs. rounds, accuracy vs. total bits transmitted, K-trajectory per client. Same seeds and capacity draws for DSFL vs ADAFL for fair comparison. | All experiment figures |
| 12–14 | Write full paper draft: intro, related work, system model, algorithm, experiments (DSFL-matched setup), conclusion. | Complete draft (6 pages) |
| 15–17 | Revise draft. Add K-trajectory visualization. Insert theoretical positioning section. | Revised draft |
| 18–19 | Final polish. IEEE two-column format. Proofread. Submit. | Submitted PDF |

---

## 9. Phase 2 Roadmap: From ADAFL to MAB-DSFL

ADAFL is deliberately designed as the first step in a two-phase research arc. The loss-feedback adaptive K in Phase 1 lays the groundwork for a full MAB formulation in Phase 2.

| Phase | Target | Deadline | Core Addition Over Previous Phase |
|-------|--------|----------|-----------------------------------|
| Phase 1: ADAFL | Globecom 2026 | April 1, 2026 | Loss-feedback adaptive K within DSFL pipeline |
| Phase 2A: Journal | IEEE ToN or TNNLS | ~Aug 2026 | Full convergence proof + regret analysis + more datasets |
| Phase 2B: MAB-DSFL | INFOCOM 2027 | July 24, 2026 | Replace feedback rule with UCB bandit + Steiger's POSS framework |
| Phase 3 (future) | IEEE JSAC or NeurIPS | 2027+ | Joint scheduling (HeteRo-Select) + compression (MAB-DSFL) |

The key design decision: ADAFL's ΔL<sub>i</sub>[*t*] signal is the natural reward signal for a MAB formulation in Phase 2B. When the Slivkins MAB framework is fully internalized, the UCB extension of ADAFL becomes a straightforward formalization of what ADAFL already does heuristically — replacing the deterministic feedback rule with a principled exploration-exploitation policy. This makes Phase 2B a genuine theoretical upgrade of Phase 1, not a different paper.

---

## 10. Connection to Prof. Ning Lu's Research Agenda

This proposal is consciously designed to land at the intersection of all three of Prof. Lu's major published contributions:

- **FL Compression Thread:** DSFL / FLAC (Globecom 2022): ADAFL is a direct extension of Mahdi's work. Prof. Lu knows the codebase, the assumptions, and the experimental setup intimately.
- **Bandit Scheduling Thread:** POSS / Backlogged Bandits (INFOCOM 2023/2024): Phase 2B maps ADAFL's feedback rule onto Steiger's bandit framework. The theoretical tools are already in the lab.
- **Wireless Systems Expertise:** Prof. Lu's background in real-time scheduling and wireless systems makes him uniquely positioned to validate the communication model and the Globecom framing of ADAFL (with scope for real-channel or time-varying models in Phase 2).

The proposed email to Prof. Lu should present ADAFL not as a new idea from outside but as the natural bridge between his two existing threads — which is precisely what it is.

---

## 11. Target Paper Outline (6-Page Globecom Format)

| Section | Content | Est. Length |
|---------|---------|-------------|
| I. Introduction | FL communication bottleneck; DSFL as prior work; static K as the gap; ADAFL as the fix; contributions | ~0.7 pages |
| II. Related Work | DSFL lineage; adaptive sparsification landscape; differentiation table (condensed) | ~0.5 pages |
| III. System Model | FL setup; communication model (same as DSFL: truncated normal capacity); DSFL pipeline recap | ~0.5 pages |
| IV. ADAFL Algorithm | Adaptive K rule; integration with DSFL Algorithm 1; pseudocode; theoretical positioning | ~1.0 pages |
| V. Experimental Results | MNIST, CIFAR-10, Speech (DSFL setting); accuracy vs. rounds; accuracy vs. bits; K-trajectory; ablation of η and τ | ~2.5 pages |
| VI. Conclusion | Summary; limitations; Phase 2 MAB extension preview | ~0.3 pages |

---

## 12. Available Resources

- **DSFL codebase** (Mahdi Beitollahi, public GitHub: github.com/mahdibeit/DSFL) — full implementation of LSS + top-K + error accumulation using Flower framework
- **HeteRo-Select codebase** — existing non-IID FL scheduling implementation; useful for optional ablations (e.g. different non-IID partitions) in Phase 2
- **Prof. Ning Lu's POSS and Backlogged Bandits papers** — theoretical scaffolding for Phase 2 MAB extension
- **Juaren Steiger's IEEE ToN 2026 paper** — most mature version of the bandit framework in the lab
- **Slivkins MAB textbook** — being read in parallel; will inform Phase 2 UCB formulation

---

*Prepared by Md Akmol Masud | March 2026 | Queen's University ECE — Incoming M.A.Sc. September 2026*
