#!/bin/bash
echo "Starting FEDS server..."
venv/bin/python3 server_feds.py > server_feds.log 2>&1 &
SERVER_PID=$!
sleep 5

echo "Starting 10 FEDS clients..."
for i in $(seq 0 9); do
    venv/bin/python3 clients/client-MNIST_feds.py --seed=$i > client_feds_$i.log 2>&1 &
done

echo "Waiting for simulation to finish. Check server_feds.log for progress."
wait $SERVER_PID
echo "Simulation complete!"
