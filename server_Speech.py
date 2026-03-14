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
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

import copy

from dsfl_feds import alastor_feds, Flatten


class _NoOpWriter:
    """No-op when TensorBoard is unavailable."""
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


def aggregateNew(results, client_id, client_metrics=None):
    """Compute weighted average with FEDS adaptive sparsification."""
    weighted_weights = [
        [layer * 1 for layer in weights] for weights, num_examples in results
    ]

    weighted_weights = [weights for _, weights in sorted(zip(client_id, weighted_weights), key=lambda pair: pair[0])]

    if client_metrics is not None:
        client_metrics = [metrics for _, metrics in sorted(zip(client_id, client_metrics), key=lambda pair: pair[0])]

    history.update(weighted_weights)

    try:
        with open("feds_history.json", "w") as fp:
            json.dump({"round": history.round}, fp)
    except Exception:
        pass

    weighted_weights_accu = copy.deepcopy(weighted_weights)
    weighted_weights = alastor_feds(weighted_weights_accu, history, client_metrics)

    number_of_users = len(results)
    weights_prime = [
        reduce(np.add, layer_updates) / number_of_users
        for layer_updates in zip(*weighted_weights)
    ]

    return weights_prime


if __name__ == "__main__":

    number_of_users = 10

    history = History()
    if SummaryWriter is not None:
        try:
            writer = SummaryWriter(comment=" FEDS - SpeechCommands - NIID")
        except Exception as e:
            print(f"TensorBoard disabled ({e})")
            writer = _NoOpWriter()
    else:
        writer = _NoOpWriter()

    class FedComp(FedAvg):

        def aggregate_evaluate(self, rnd, results, failures):
            if not results:
                return None

            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
            loss = [r.loss * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]

            accuracy_aggregated = sum(accuracies) / sum(examples)
            loss_aggregated = sum(loss) / sum(examples)
            print(f"Round {rnd} accuracy: {accuracy_aggregated}")
            print(f"Round {rnd} loss: {loss_aggregated}")
            writer.add_scalar('accuracy_aggregated', accuracy_aggregated, rnd)
            writer.add_scalar('loss_aggregated', loss_aggregated, rnd)

            try:
                if os.path.exists('feds_k_tracker.json'):
                    with open('feds_k_tracker.json', 'r') as f:
                        k_tracker = json.load(f)
                        k_list = k_tracker.get('k_list', [])
                        if k_list:
                            k_array = np.array(k_list)
                            writer.add_scalar('feds/k_mean', k_array.mean(), rnd)
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
            client_metrics = [
                dict(fit_res.metrics)
                for client, fit_res in results
            ]
            return ndarrays_to_parameters(aggregateNew(weights_results, client_id, client_metrics)), {}

    # Load Speech Commands initial model
    with open('initial_global_model_Speech', 'rb') as infile:
        initial_model = pickle.load(infile)

    history.updateGlobal(Flatten(initial_model[0][0])[0])
    history.updateError([[0]*len(Flatten(initial_model[0][0])[0]) for _ in range(number_of_users)])

    strategy = FedComp(
        fraction_fit=1,
        fraction_eval=1,
        min_fit_clients=number_of_users,
        min_available_clients=number_of_users,
        initial_parameters=ndarrays_to_parameters(initial_model[0][0]),
    )

    flwr.server.start_server(
        server_address="localhost:8080",
        config={"num_rounds": 300},
        strategy=strategy,
    )
