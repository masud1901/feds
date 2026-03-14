# FEDS configuration reference

## Server (server.py)
- number_of_users: 10 (line ~103)
- num_rounds: 300 (line ~207)
- server_address: localhost:8080 (line ~206)
- initial_global_model_MNIST: file in CWD (line ~184)
- fraction_fit, fraction_eval: 1 (lines ~193-194)
- min_fit_clients, min_available_clients: number_of_users (lines ~195-196)
- TensorBoard: SummaryWriter comment at line ~106, logs in runs/

## FEDS (utils/LayerWiseSparsification_feds.py)
- K_min: max(100, d//100) (~43)
- K_max: d (~44)
- WARMUP_ROUNDS: 5 (~45)
- MOMENTUM: 0.3 (~46)
- ETA_BASE: 1000000.0 (~47)
- ETA_DECAY: 0.995 (~48)
- k_tracker_file: feds_k_tracker.json (~51)
- Initial K file at line ~63; create with test_feds_logic.py or scripts/create_artifacts_colab.py
- History cap: 100 (~153-155)
- tau: computed each round from mean relative loss improvement

## MNIST client (clients/client-MNIST.py)
- DEVICE: cpu (~28), use cuda:0 for GPU
- lr: 0.1 (~63)
- epochs per round: 1 (~196)
- batch_size: 32 (~133, ~138)
- IID: False (~154)
- Data path: ../data/mnist/ (~130)
- Server: localhost:8080 (~210)
- --seed: CLI (~157), typically 0..9

## Artifacts (repo root)
- Initial K: create with test_feds_logic.py or scripts/create_artifacts_colab.py
- initial_global_model_MNIST: create with scripts/create_artifacts_colab.py

## Runtime
- PYTHONPATH: include utils when running server
- CWD: repo root for server and clients

## Colab
- REPO in first cell; PYTHONPATH=utils for server; 10 clients seeds 0..9; 300 rounds default
