# FEDS: Loss-Feedback Adaptive Sparsification for Federated Learning

This repository implements **FEDS**, a loss-feedback driven mechanism for adaptive sparsification in federated learning over heterogeneous wireless networks. FEDS extends Mahdi Beitollahi's **DSFL (Dynamic Sparsification for Federated Learning)** codebase with dynamic compression rates that adapt based on local training progress.

## Key Features

- **Two-level sparsification pipeline** (inherited from DSFL):
  - **Layer-wise Similarity Sparsification (LSS)** using CKA to exploit global redundancy across clients
  - **Top-K sparsification** to respect each client's uplink capacity

- **FEDS adaptive K mechanism** (this work):
  - Loss-feedback driven per-client sparsification rate updates
  - **Zero additional communication overhead** - uses only locally available training signals
  - **Momentum-smoothed updates** for stability
  - **Z-score normalization** of loss improvements for dataset-agnostic hyperparameters
  - **Warmup phase** to prevent early-round instability
  - **Decaying learning rate** for K updates (eta)

The update rule:
```
K_i[t+1] = clip(K_i[t] - η * normalized(ΔL_i[t] - τ), K_min, K_max)
```
where `ΔL_i[t]` is the relative loss improvement, `τ` is the mean improvement across clients, and normalization makes `η` work across different datasets.

## Project Structure

```
adaptive-sparsification/
├── server.py                          # Flower server with FedComp strategy
├── clients/
│   ├── client-MNIST.py                # MNIST client (CNN)
│   ├── client-CIFAR.py                # CIFAR-10 client (CNN)
│   └── client-SpeechCommands.py       # Speech Commands client (CNN)
├── utils/
│   ├── dsfl.py                        # Original DSFL utilities
│   ├── dsfl_feds.py                   # FEDS-specific utilities
│   ├── LayerWiseSparsification.py     # Original DSFL sparsification
│   ├── LayerWiseSparsification_feds.py # FEDS adaptive sparsification
│   ├── FindingK.py                    # CKA analysis for CIFAR
│   ├── FindingKSpeech.py              # CKA analysis for Speech
│   ├── visualize_k_trajectory.py      # K trajectory visualization
│   └── validate_feds.py               # FEDS validation script
├── docs/
│   └── feds.md                        # Research proposal & Globecom plan
├── run-mnistclients.sh                # Launch MNIST clients
├── run-cifarclients.sh                # Launch CIFAR clients
└── run-speechclients.sh               # Launch Speech clients
```

## Installation

Create a Python environment and install dependencies:

```bash
pip install flwr torch torchvision torchaudio tensorboard numpy matplotlib scipy
```

For CKA-based analysis:
```bash
pip install torch_cka
```

## Running Experiments

### 1. Start the Server

```bash
python server.py
```

The server will:
- Initialize with a pre-trained global model (`initial_global_model_MNIST`)
- Run for 300 rounds by default
- Log accuracy, loss, and K statistics to TensorBoard

### 2. Start Clients

Run clients in separate terminals:

```bash
# For MNIST
./run-mnistclients.sh

# For CIFAR-10
./run-cifarclients.sh

# For Speech Commands
./run-speechclients.sh
```

### 3. Monitor Training

View TensorBoard logs:
```bash
tensorboard --logdir=runs/
```

Logged metrics include:
- `accuracy_aggregated`, `loss_aggregated` - global model performance
- `feds/k_mean`, `feds/k_std` - K value statistics
- `feds/k_client_{i}` - per-client K values

### 4. Analyze K Trajectory

After training, visualize the adaptive K behavior:

```bash
python utils/visualize_k_trajectory.py --input feds_k_trajectory.json --output figures/
```

This generates:
- `feds_k_trajectory.png` - Per-client K values over rounds
- `feds_loss_k_correlation.png` - Loss improvement vs K change correlation
- `feds_communication_savings.png` - Communication savings over time

### 5. Validate FEDS

Verify the adaptive mechanism is working:

```bash
python utils/validate_feds.py
```

## Configuration

Key FEDS parameters in `utils/LayerWiseSparsification_feds.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K_min` | `max(100, d//100)` | Minimum K (at least 1% of parameters) |
| `K_max` | `d` | Maximum K (all parameters) |
| `WARMUP_ROUNDS` | `5` | Static K rounds before adaptation starts |
| `MOMENTUM` | `0.3` | EMA smoothing factor for K updates |
| `ETA_BASE` | `1,000,000` | Base learning rate for K updates |
| `ETA_DECAY` | `0.995` | Per-round decay factor for eta |

## Output Files

| File | Description |
|------|-------------|
| `feds_k_tracker.json` | Current K values, loss history, K history |
| `feds_k_trajectory.json` | Full K trajectory for visualization |
| `feds_history.json` | Server round tracking |
| `runs/` | TensorBoard logs |

## Research Context

This work targets **IEEE Globecom 2026** and directly extends DSFL by replacing static K assignment with loss-feedback adaptation. See `docs/feds.md` for:
- Full research proposal
- Related work positioning
- Experimental plan
- Phase 2 roadmap (MAB extension for INFOCOM 2027)

## Comparison with DSFL

| Aspect | DSFL (Original) | FEDS (This Work) |
|--------|-----------------|------------------|
| K selection | Truncated normal distribution | Loss-feedback adaptive |
| Hyperparameters | α, γ require per-dataset tuning | Single η with normalization |
| K stability | Random sampling each round | Momentum-smoothed updates |
| Early rounds | No special handling | Warmup phase |
| Logging | Basic metrics | K trajectory + TensorBoard |

## Citation

```bibtex
@inproceedings{masud2026feds,
  title={Loss-Feedback Adaptive Sparsification for Communication-Efficient Federated Learning over Heterogeneous Wireless Networks},
  author={Masud, Md Akmol and Lu, Ning},
  booktitle={IEEE GLOBECOM},
  year={2026}
}
```

## License

Based on the original DSFL implementation by Mahdi Beitollahi. FEDS extensions follow the same license.
