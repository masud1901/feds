"""
FEDS Server — fixed for Flower >= 1.0
"""

import flwr
from typing import Dict, List, Optional, Tuple
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes,
    EvaluateRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from functools import reduce
import numpy as np
import json
import os
import warnings
import copy

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

import sys
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if os.path.join(REPO, "common") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "common"))

from common.flatten import Flatten
from methods.feds.aggregation import alastor_feds

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# No-op TensorBoard writer (graceful fallback for Colab protobuf conflicts)
# =============================================================================
class _NoOpWriter:
    def add_scalar(self, *args, **kwargs):
        pass


# =============================================================================
# History tracker
# =============================================================================
class History:
    def __init__(self):
        self.list = []
        self.error = []
        self.globals = []
        self.round = 0

    def update(self, weighted_weights):
        self.list.append(copy.deepcopy(weighted_weights))
        self.round += 1
        print(f"\nGlobal Round {self.round}")

    def updateError(self, errors):
        self.error.append(errors[:])

    def updateGlobal(self, globmodel):
        self.globals.append(globmodel[:])

    def Len(self):
        print(len(self.list))


# =============================================================================
# Aggregation with FEDS adaptive sparsification
# =============================================================================
def aggregateNew(
    results: List[Tuple[List[np.ndarray], int]],
    client_id: List[int],
    client_metrics=None,
) -> List[np.ndarray]:

    num_examples_total = sum(n for _, n in results)

    weighted_weights = [[layer * 1 for layer in weights] for weights, _ in results]

    # Sort by client id
    weighted_weights = [
        w for _, w in sorted(zip(client_id, weighted_weights), key=lambda p: p[0])
    ]
    if client_metrics is not None:
        client_metrics = [
            m for _, m in sorted(zip(client_id, client_metrics), key=lambda p: p[0])
        ]

    history.update(weighted_weights)

    # Persist round number for notebook monitoring
    try:
        with open("feds_history.json", "w") as fp:
            json.dump({"round": history.round}, fp)
    except Exception:
        pass

    # FEDS adaptive sparsification
    weighted_weights_accu = copy.deepcopy(weighted_weights)
    weighted_weights = alastor_feds(weighted_weights_accu, history, client_metrics)

    # Aggregate (simple average)
    weights_prime: List[np.ndarray] = [
        reduce(np.add, layer_updates) / number_of_users
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


# =============================================================================
# Custom FedAvg strategy
# =============================================================================
class FedComp(FedAvg):

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        if not results:
            return None, {}

        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        losses     = [r.loss * r.num_examples                 for _, r in results]
        examples   = [r.num_examples                          for _, r in results]

        accuracy_aggregated = sum(accuracies) / sum(examples)
        loss_aggregated     = sum(losses)     / sum(examples)

        print(f"Round {server_round} accuracy: {accuracy_aggregated:.4f}  loss: {loss_aggregated:.4f}")
        writer.add_scalar("accuracy_aggregated", accuracy_aggregated, server_round)
        writer.add_scalar("loss_aggregated",     loss_aggregated,     server_round)

        # Log FEDS K statistics
        try:
            if os.path.exists("feds_k_tracker.json"):
                with open("feds_k_tracker.json") as f:
                    k_tracker = json.load(f)
                k_list = k_tracker.get("k_list", [])
                if k_list:
                    k_array = np.array(k_list)
                    writer.add_scalar("feds/k_mean", k_array.mean(), server_round)
                    writer.add_scalar("feds/k_std",  k_array.std(),  server_round)
                    for i in range(min(5, len(k_list))):
                        writer.add_scalar(f"feds/k_client_{i}", k_list[i], server_round)
        except Exception as e:
            print(f"Warning: Could not log K stats: {e}")

        # FIX: pass server_round (not rnd) to super — Flower >= 1.0 signature
        return super().aggregate_evaluate(server_round, results, failures)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        client_id = [int(client.cid[11:]) for client, _ in results]

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        client_metrics = [dict(fit_res.metrics) for _, fit_res in results]

        aggregated = aggregateNew(weights_results, client_id, client_metrics)
        return ndarrays_to_parameters(aggregated), {}


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":

    number_of_users = 10
    DATASET = os.environ.get("FEDS_DATASET", "MNIST")

    model_files = {
        "MNIST":   "initial_global_model_MNIST.npz",
        "CIFAR10": "initial_global_model_CIFAR.npz",
        "Speech":  "initial_global_model_Speech.npz",
    }
    model_file = model_files.get(DATASET, "initial_global_model_MNIST.npz")

    history = History()

    # TensorBoard — graceful fallback
    if SummaryWriter is not None:
        try:
            writer = SummaryWriter(comment=f" FEDS - {DATASET} - NIID")
        except Exception as e:
            print(f"TensorBoard disabled ({e})")
            writer = _NoOpWriter()
    else:
        writer = _NoOpWriter()

    # Load initial model
    print(f"Loading model: {model_file}")
    data = np.load(model_file)
    keys = sorted(data.files, key=lambda k: int(k.split("_")[1]) if "_" in k else 0)
    params = [data[k] for k in keys]
    history.updateGlobal(Flatten(params)[0])
    history.updateError([[0] * len(Flatten(params)[0]) for _ in range(number_of_users)])

    strategy = FedComp(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=number_of_users,
        min_available_clients=number_of_users,
        initial_parameters=ndarrays_to_parameters(params),
    )

    print(f"Starting FEDS server for {DATASET}")

    # FIX: ServerConfig replaces plain dict in Flower >= 1.0
    flwr.server.start_server(
        server_address="localhost:8080",
        config=flwr.server.ServerConfig(num_rounds=300),
        strategy=strategy,
    )