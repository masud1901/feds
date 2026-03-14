# FEDS: Loss-Feedback Adaptive Sparsification for Federated Learning

This repository contains a modified version of Mahdi Beitollahi’s **DSFL (Dynamic Sparsification for Federated Learning)** codebase, extended with **FEDS**, a loss-feedback driven mechanism for adaptive sparsification in heterogeneous wireless FL.

- **DSFL baseline**: two-level sparsification with
  - **Layer-wise Similarity Sparsification (LSS)** using CKA to exploit global redundancy across clients, and
  - **Extended top-K sparsification** to respect each client’s uplink capacity.
- **FEDS extension (this work)**: keeps DSFL’s system model and LSS + top‑K pipeline, but replaces the *static*, truncated-normal choice of sparsification rate \(K_i[t]\) with a **loss-feedback adaptive rule** on each client:
  \(K_i[t+1] = \text{clip}\big(K_i[t] + \eta(\Delta L_i[t] - \tau), K_{\min}, K_{\max}\big)\),
  where \(\Delta L_i[t]\) is the client’s local loss improvement. This requires **no additional communication** and targets better accuracy-per-bit under heterogeneous capacities.

The high-level research proposal and Globecom positioning are documented in `docs/feds.md`.

## Project Structure

- `server.py`: Flower server with a custom `FedComp` strategy and the DSFL/FEDS aggregation hook.
- `clients/`:
  - `client-MNIST.py`
  - `client-CIFAR.py`
  - `client-SpeechCommands.py`  
  Standard Flower NumPy clients that perform local SGD on each dataset.
- `utils/`:
  - `dsfl.py`: flatten/de-flatten utilities and the `alastor` hook which calls the sparsification logic.
  - `LayerWiseSparsification.py`: implementation of LSS, top‑K sparsification, and error accumulation (DSFL core).
  - `FindingK.py`, `FindingKSpeech.py`: CKA-based layer similarity tools used to build LSS masks.
- `docs/feds.md`: FEDS proposal, related work, system model, experimental plan, and conference roadmap.

## Installation

Create a Python environment (recommended) and install the main dependencies:

```bash
pip install flwr torch torchvision torchaudio tensorboard numpy matplotlib scipy
```

You may also need to install `torch_cka` and dataset-specific dependencies used in `utils/FindingK*.py`.

## Running Experiments

The workflow follows the original DSFL setup (Flower-based FL with multiple clients):

1. **Start the server** (from the repo root):

   ```bash
   python server.py
   ```

2. **Start clients** in separate terminals using the provided scripts (one per client/device). For example, to run MNIST:

   ```bash
   ./run-mnistclients.sh
   ```

   Similarly, use:

   - `./run-cifarclients.sh` for CIFAR‑10,
   - `./run-speechclients.sh` for Speech Commands.

The current codebase is being adapted from DSFL to FEDS; for strict DSFL reproduction vs. FEDS comparisons (as planned for the Globecom submission), see the details and parameter choices in `docs/feds.md`.

## Status

- **DSFL functionality**: imported from the original code and under test in this repository.
- **FEDS integration**: in progress. The goal is to:
  - implement the loss-feedback \(K_i[t]\) update rule,
  - plug it into the existing LSS + top‑K pipeline,
  - and reproduce DSFL’s experimental setup for MNIST, CIFAR‑10, and Speech Commands.

Once the adaptive K logic is fully wired, this README will be updated with precise configuration flags and plotting scripts for reproducing the FEDS vs. DSFL figures.

## License

Based on the original DSFL implementation; see the upstream license and paper for details. Any new FEDS-specific extensions in this repository follow the same license unless otherwise noted.
