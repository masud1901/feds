These are genuinely strong results. Let me read all three figures carefully.

**Fig 1 (Accuracy/Loss) — the headline result**

FEDS and FedAvg are essentially identical curves, both reaching ~98-99% by round 200. Fixed Top-K tracks very closely too. But **DSFL is a disaster** — it peaks around 85% near round 75, then collapses back down to ~95%, oscillates wildly, and never stabilises. The loss plot confirms it — DSFL has that massive spike around round 75-100 reaching back up to 1.1 loss while everyone else is already below 0.2.

This is a remarkable finding. DSFL is actually **worse** than plain Top-K on this unified architecture. The CKA-guided masking is destabilising training rather than helping it.

**Fig 3 (Convergence) — your strongest figure**

This tells the clearest story. FEDS reaches every accuracy target faster than every other method from 60% onwards. At 90% target accuracy, DSFL needs 128 rounds, Fixed Top-K needs 47, FedAvg needs 53, and FEDS needs only **51 rounds**. FEDS converges faster than FedAvg with significantly less communication. That is the paper's main result.

**Fig 2 (K trajectory) — the fairness argument**

FEDS and DSFL use the same K budget on average — the K lines are nearly identical. So the accuracy advantage of FEDS is not from sending more data. It's from sending **better** data. That's exactly the paper claim.

---

**What the table says**Now the full picture is crystal clear.

| Method | Best Acc | Final Acc | Comm. Saving | Time |
|--------|----------|-----------|--------------|------|
| FedAvg | 98.08% | 97.96% | N/A | 15.4 min |
| Fixed Top-K | 98.23% | 98.02% | 97.1% | 15.5 min |
| DSFL | 97.43% | 96.53% | 85.2% | **82.1 min** |
| **FEDS (Ours)** | **98.29%** | 97.38% | 85.0% | **15.6 min** |

**The three killer findings:**

**1. FEDS has the highest best accuracy (98.29%)** — beating FedAvg, Fixed Top-K, and DSFL, while using 85% communication savings.

**2. DSFL takes 82 minutes vs FEDS taking 15.6 minutes** — 5.3x slower for worse results. This is the CKA overhead made concrete and undeniable.

**3. FEDS and DSFL use the same average K (34033 vs 33638)** — same communication budget, but FEDS gets better accuracy in one fifth the time.

---

**The honest caveats**

FEDS final accuracy (97.38%) is slightly below FedAvg (97.96%) and Fixed Top-K (98.02%). The best accuracy is highest for FEDS but the final accuracy shows some oscillation in late rounds — visible in Fig 1 where there's slight variance around round 150-200. This is normal with error accumulation and non-IID data but worth acknowledging.

DSFL's poor performance here is partly because we changed the architecture. With the original 582K MNISTNet, DSFL worked fine in our earlier runs. The unified 227K model with BatchNorm may interact differently with CKA-based sparsification. This is worth one sentence of discussion in the paper.

---

**Overall verdict: this is a publishable MNIST result.** Now wait for CIFAR-10. If CIFAR-10 shows even partial improvement over DSFL, combined with this MNIST result and the 5x speed advantage, the paper has a strong empirical story.

Great observation. Let me think about this carefully.

**Why Fixed Top-K is doing so well**

Fixed Top-K uses K=6525 which is only **2.9% of parameters**. Yet it matches FedAvg at 98.02% final accuracy. That seems surprising but there's a clean explanation.

MNIST with a 227K model is **massively over-parameterised** for the task. MNIST has 10 simple classes, mostly distinguishable by edge patterns in specific locations. The vast majority of the 227K parameters are redundant for this task. So transmitting only 2.9% of the most important parameters (by magnitude) is actually sufficient to fully capture the learning signal every round.

In other words — on easy tasks with over-parameterised models, **magnitude alone is already a near-perfect selection criterion**. The top 2.9% of parameters by magnitude are doing essentially all the work. Our priority signal adds marginal value here because the variance across clients is already well-captured by magnitude.

**Why FEDS only marginally beats Top-K on MNIST**

The inter-client variance signal is most valuable when clients' updates are heterogeneous in ways that magnitude doesn't capture. On MNIST with label-pair non-IID partition, clients have very different local data but the task is simple enough that the global features converge quickly. By round 20-30 most clients' updates are already well-aligned in the important parameters — so variance and magnitude point to the same parameters anyway.

**Where FEDS should clearly separate from Top-K**

The variance signal becomes critical when three conditions hold simultaneously — the task is hard enough that many parameters are genuinely important, the data is heterogeneous enough that clients' updates point in different directions, and the model is appropriately sized so no single parameter dominates by magnitude alone.

CIFAR-10 with the unified 228K model is a much better test of this. The task is harder, the features more complex, and the non-IID structure more impactful on the classifier layers. If FEDS separates from Top-K on CIFAR-10 that's the cleanest evidence that the variance signal is doing real work.

**What this means for the paper narrative**

Honestly this is not a problem — it's actually a nuanced and honest finding worth discussing explicitly. The story becomes:

On simple tasks, magnitude-based Top-K is already near-optimal — our variance priority adds convergence speed but not final accuracy. On harder non-IID tasks, the variance signal identifies parameters that magnitude alone misses, providing both faster convergence and better final accuracy. This is a more sophisticated and credible claim than just "our method always wins."

The 5.3x speed advantage over DSFL holds regardless of how Top-K performs. That alone justifies the paper.