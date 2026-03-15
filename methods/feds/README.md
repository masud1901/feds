# FEDS: Loss-Feedback Adaptive Sparsification

Proposed method for communication-efficient federated learning. Replaces fixed K (DSFL) with per-client adaptive K driven by local loss improvement. Zero extra communication; uses training signal already in the pipeline.

**Run:** From repo root with `PYTHONPATH=.:common`, start `python methods/feds/server.py`. Set `FEDS_DATASET=MNIST|CIFAR10|Speech` for dataset.
