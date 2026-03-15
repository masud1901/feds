#!/bin/bash
# Run FedAvg baseline. From repo root. Usage: FEDS_DATASET=MNIST ./scripts/run_fedavg.sh
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}:${REPO}/common:${REPO}/utils"
DATASET="${FEDS_DATASET:-MNIST}"
echo "Starting FedAvg server for $DATASET..."
python methods/fedavg/server.py &
SERVER_PID=$!
sleep 5
echo "Starting 10 clients..."
for i in $(seq 0 9); do
  if [ "$DATASET" = "MNIST" ]; then
    python clients/mnist.py --seed=$i &
  elif [ "$DATASET" = "CIFAR10" ]; then
    python clients/cifar10.py --seed=$i &
  else
    python clients/speech_commands.py --seed=$i &
  fi
done
wait $SERVER_PID
