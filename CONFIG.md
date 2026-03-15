# FEDS configuration reference

## Servers (run from repo root with PYTHONPATH=.:common:utils)

- **methods/fedavg/server.py** — FedAvg baseline
- **methods/dsfl/server.py** — DSFL (fixed K)
- **methods/feds/server.py** — FEDS (adaptive K)

Common: number_of_users=10, num_rounds=300, server_address=localhost:8080, fraction_fit/fraction_eval=1, min_fit_clients/min_available_clients=number_of_users. Set **FEDS_DATASET** to MNIST, CIFAR10, or Speech. Initial model files: initial_global_model_MNIST.npz, initial_global_model_CIFAR.npz, initial_global_model_Speech.npz (NumPy compressed). K initial values: K_initial.json, K_Speech_initial.json (JSON).

## FEDS (methods/feds/sparsification.py)

- K_min: max(100, d//100), K_max: d
- WARMUP_ROUNDS: 5, MOMENTUM: 0.3, ETA_BASE: 1000000.0, ETA_DECAY: 0.995
- k_tracker_file: feds_k_tracker.json; initial K from K_initial.json or K_Speech_initial.json
- Create artifacts: scripts/create_artifacts.py (writes .npz models and JSON K files)

## Clients (clients/mnist.py, cifar10.py, speech_commands.py)

- DEVICE: cpu or cuda:0, lr and batch_size per client script
- IID: False (non-IID). Data paths: ../data/mnist/, etc. Server: localhost:8080. --seed: 0..9

## Scripts

- scripts/run_fedavg.sh, run_dsfl.sh, run_feds.sh — run server + 10 clients
- scripts/create_artifacts.py --dataset all — create initial models and K pickles
- scripts/visualize_k_trajectory.py — FEDS K trajectory figures

## Runtime

- CWD: repo root. PYTHONPATH=.:common:utils (and utils for DSFL/FindingK).

## Outputs

- TensorBoard: runs/ (or outputs/runs/)
- results_FedAvg_*.json, results_DSFL_*.json, feds_k_tracker.json, feds_k_trajectory.json
