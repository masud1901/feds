"""FedAvg Baseline Server - No sparsification, standard federated averaging."""
import flwr
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
import pickle
import json
import os
import warnings

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

warnings.filterwarnings("ignore", category=UserWarning)


class _NoOpWriter:
    def add_scalar(self, *args, **kwargs):
        pass


if __name__ == "__main__":

    number_of_users = 10
    DATASET = os.environ.get("FEDS_DATASET", "MNIST")

    model_files = {
        "MNIST": "initial_global_model_MNIST",
        "CIFAR10": "initial_global_model_CIFAR",
        "Speech": "initial_global_model_Speech"
    }
    model_file = model_files.get(DATASET, "initial_global_model_MNIST")

    if SummaryWriter is not None:
        try:
            writer = SummaryWriter(comment=f" FedAvg - {DATASET}")
        except Exception:
            writer = _NoOpWriter()
    else:
        writer = _NoOpWriter()

    class FedAvgBaseline(FedAvg):
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

            results_file = f"results_FedAvg_{DATASET}.json"
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

    print(f"Loading model: {model_file}")
    with open(model_file, 'rb') as f:
        initial_model = pickle.load(f)

    strategy = FedAvgBaseline(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=number_of_users,
        min_available_clients=number_of_users,
        initial_parameters=ndarrays_to_parameters(initial_model[0][0]),
    )

    print(f"Starting FedAvg baseline for {DATASET}")
    flwr.server.start_server(
        server_address="localhost:8080",
        config={"num_rounds": 300},
        strategy=strategy,
    )
