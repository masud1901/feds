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
from torch.utils.tensorboard import SummaryWriter
import copy

from dsfl_feds import alastor_feds, Flatten

warnings.filterwarnings("ignore", category=UserWarning)

class History():
    def __init__(self):
        self.list = []
        self.error = []
        self.globals = []
        self.round = 0
        
    def update(self, weighted_weights):
        copied_list=copy.deepcopy(weighted_weights)
        self.list.append(copied_list)
        self.round +=1
        
        print('\n', "Gobal Round", self.round)
        
    def updateError(self, errors):
        self.error.append(errors[:])
        
    def updateGlobal(self, globmodel):
        self.globals.append(globmodel[:])
        
    def Len(self):
        
        print(len(self.list))
    
    each_round_weighted_weights=[]
   
    

def aggregateNew(results: List[Tuple[List[np.ndarray], int]], client_id, client_metrics=None) -> List[np.ndarray]:
    """Compute weighted average with FEDS adaptive sparsification."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * 1 for layer in weights] for weights, num_examples in results
    ]

    # Sort and match the id with weights
    weighted_weights = [weights for _, weights in sorted(zip(client_id, weighted_weights), key=lambda pair: pair[0])]

    # Sort metrics to match client order if provided
    if client_metrics is not None:
        client_metrics = [metrics for _, metrics in sorted(zip(client_id, client_metrics), key=lambda pair: pair[0])]

    history.update(weighted_weights)
    # Save history for debugging (using JSON-safe format when possible)
    try:
        import json
        with open("feds_history.json", "w") as fp:
            json.dump({"round": history.round}, fp)
    except Exception:
        pass

    """ Saved File:
        First Index: Round
        Second Index: User
        Third Index: Layer
    """

    """FEDS adaptive sparsification with loss feedback"""
    weighted_weights_accu = copy.deepcopy(weighted_weights)
    weighted_weights = alastor_feds(weighted_weights_accu, history, client_metrics)

    """Aggregate"""
    weights_prime: List[np.ndarray] = [
        reduce(np.add, layer_updates) / number_of_users  # num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]

    return weights_prime



if __name__ == "__main__":

    number_of_users=10
    
    history=History()
    writer = SummaryWriter(comment=" FEDS - MNIST - NIID - Adaptive K with Loss Feedback")

    #Extend class FedAVG
    class FedComp(FedAvg):
        
        """Save and graph the aggregated loss and accuracies, including FEDS K statistics"""
        def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
        ) -> Optional[float]:
            """Aggregate evaluation losses using weighted average."""
            if not results:
                return None

            # Weigh accuracy of each client by number of examples used
            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
            loss = [r.loss * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]

            # Aggregate and print custom metric
            accuracy_aggregated = sum(accuracies) / sum(examples)
            loss_aggregated = sum(loss) / sum(examples)
            print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")
            print(f"Round {rnd} loss aggregated from client results: {loss_aggregated}")
            writer.add_scalar('accuracy_aggregated', accuracy_aggregated, rnd)
            writer.add_scalar('loss_aggregated', loss_aggregated, rnd)

            # Log FEDS K statistics to TensorBoard
            try:
                if os.path.exists('feds_k_tracker.json'):
                    with open('feds_k_tracker.json', 'r') as f:
                        k_tracker = json.load(f)
                        k_list = k_tracker.get('k_list', [])
                        if k_list:
                            k_array = np.array(k_list)
                            writer.add_scalar('feds/k_mean', k_array.mean(), rnd)
                            writer.add_scalar('feds/k_std', k_array.std(), rnd)
                            writer.add_scalar('feds/k_min', k_array.min(), rnd)
                            writer.add_scalar('feds/k_max', k_array.max(), rnd)

                            # Per-client K values (first 5 clients for readability)
                            for i in range(min(5, len(k_list))):
                                writer.add_scalar(f'feds/k_client_{i}', k_list[i], rnd)
            except Exception as e:
                print(f"Warning: Could not log K statistics: {e}")

            # Call aggregate_evaluate from base class (FedAvg)
            return super().aggregate_evaluate(rnd, results, failures)
                
        def aggregate_fit(
                self,
                rnd: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[BaseException],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                """Aggregate fit results using weighted average with FEDS."""
                client_id = [int(client.cid[11:]) for client, _ in results]
                if not results:
                    return None, {}
                # Do not aggregate if there are failures and failures are not accepted
                if not self.accept_failures and failures:
                    return None, {}
                # Convert results and extract metrics
                weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    for client, fit_res in results
                ]
                # Extract per-client metrics (train_loss for FEDS adaptive K)
                client_metrics = [
                    dict(fit_res.metrics)
                    for client, fit_res in results
                ]
                return ndarrays_to_parameters(aggregateNew(weights_results, client_id, client_metrics)), {}
    

    # Set the initial model for reproducability
    infile = open('initial_global_model_MNIST','rb')
    initial_model = pickle.load(infile)
    
    history.updateGlobal(Flatten(initial_model[0][0])[0])
    
    history.updateError([[0]*len(Flatten(initial_model[0][0])[0]) for _ in range(number_of_users)])
    
    # Define strategy (initial_parameters must be Parameters type in current Flower API)
    strategy = FedComp(
        fraction_fit=1,
        fraction_eval=1,
        min_fit_clients=number_of_users,
        min_available_clients=number_of_users,
        initial_parameters=ndarrays_to_parameters(initial_model[0][0]),
    )


    
    
    # Start server
    flwr.server.start_server(
        server_address="localhost:8080",
        config={"num_rounds": 300},
        strategy=strategy,
    )