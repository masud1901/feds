"""DSFL Server - Original Dynamic Sparsified Federated Learning with fixed K."""
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
import pickle
import json
import os
import warnings
import copy
import sys

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

sys.path.insert(0, 'utils')
from dsfl import alastor, Flatten


class _NoOpWriter:
    def add_scalar(self, *args, **kwargs):
        pass

warnings.filterwarnings("ignore", category=UserWarning)


class History():
    def __init__(self):
        self.list = []
        self.error = []
        self.globals = []
        self.round = 0

    def update(self, weighted_weights):
        copied_list = copy.deepcopy(weighted_weights)
        self.list.append(copied_list)
        self.round += 1
        print('\n', "Global Round", self.round)

    def updateError(self, errors):
        self.error.append(errors[:])

    def updateGlobal(self, globmodel):
        self.globals.append(globmodel[:])


def aggregateDSFL(results, client_id):
    """Compute weighted average with DSFL (fixed K sparsification)."""
    weighted_weights = [
        [layer * 1 for layer in weights] for weights, num_examples in results
    ]

    weighted_weights = [weights for _, weights in sorted(zip(client_id, weighted_weights), key=lambda pair: pair[0])]
    history.update(weighted_weights)

    try:
        with open("dsfl_history.json", "w") as fp:
            json.dump({"round": history.round}, fp)
    except Exception:
        pass

    weighted_weights = alastor(weighted_weights, history)

    number_of_users = len(results)
    weights_prime = [
        reduce(np.add, layer_updates) / number_of_users
        for layer_updates in zip(*weighted_weights)
    ]

    return weights_prime


if __name__ == "__main__":

    number_of_users = 10
    DATASET = os.environ.get("FEDS_DATASET", "MNIST")

    model_files = {
        "MNIST": "initial_global_model_MNIST",
        "CIFAR10": "initial_global_model_CIFAR",
        "Speech": "initial_global_model_Speech"
    }
    model_file = model_files.get(DATASET, "initial_global_model_MNIST")

    history = History()

    if SummaryWriter is not None:
        try:
            writer = SummaryWriter(comment=f" DSFL - {DATASET}")
        except Exception:
            writer = _NoOpWriter()
    else:
        writer = _NoOpWriter()

    class DSFLStrategy(FedAvg):
        def aggregate_evaluate(self, rnd, results, failures):
            if not results:
                return None

            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
            losses = [r.loss * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]

            accuracy_aggregated = sum(accuracies) / sum(examples)
            loss_aggregated = sum(losses) / sum(examples)
            print(f"Round {rnd} accuracy: {accuracy_aggregated:.4f}")
            print(f"Round {rnd} loss: {loss_aggregated:.4f}")
            writer.add_scalar('accuracy', accuracy_aggregated, rnd)
            writer.add_scalar('loss', loss_aggregated, rnd)

            results_file = f"results_DSFL_{DATASET}.json"
            try:
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        all_results = json.load(f)
                else:
                    all_results = {"rounds": [], "accuracy": [], "loss": []}
                all_results["rounds"].append(rnd)
                all_results["accuracy"].append(accuracy_aggregated)
                all_results["loss"].append(loss_aggregated)
                with open(results_file, 'w') as f:
                    json.dump(all_results, f)
            except Exception:
                pass

            return super().aggregate_evaluate(rnd, results, failures)

        def aggregate_fit(self, rnd, results, failures):
            client_id = [int(client.cid[11:]) for client, _ in results]
            if not results:
                return None, {}
            if not self.accept_failures and failures:
                return None, {}
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for client, fit_res in results
            ]
            return ndarrays_to_parameters(aggregateDSFL(weights_results, client_id)), {}

    print(f"Loading model: {model_file}")
    with open(model_file, 'rb') as f:
        initial_model = pickle.load(f)

    history.updateGlobal(Flatten(initial_model[0][0])[0])
    history.updateError([[0]*len(Flatten(initial_model[0][0])[0]) for _ in range(number_of_users)])

    strategy = DSFLStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=number_of_users,
        min_available_clients=number_of_users,
        initial_parameters=ndarrays_to_parameters(initial_model[0][0]),
    )

    print(f"Starting DSFL server for {DATASET}")
    flwr.server.start_server(
        server_address="localhost:8080",
        config={"num_rounds": 300},
        strategy=strategy,
    )
