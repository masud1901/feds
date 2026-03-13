import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock findingK out to avoid torch dependency
import sys
import types
module = types.ModuleType('FindingKSpeech')
module.findK = lambda: [0.1]*10  # dummy return
sys.modules['FindingKSpeech'] = module

from utils.LayerWiseSparsification_adafl import layerSparsification

class History:
    def __init__(self):
        self.round = 2
        # Create 5 rounds of history, each round has 10 users with 1000 parameters
        self.error = [[np.zeros(1000) for _ in range(10)] for _ in range(5)]
        self.globals = [np.zeros(1000) for _ in range(5)]  # globals is actually list of global models per round
        
    def updateError(self, error):
        self.error.append(error)
        
    def updateGlobal(self, glob):
        self.globals.append(glob)

# Create dummy history
history = History()
weights = [np.random.normal(0, 1, 1000) for _ in range(10)]
oldseperation = [0, 1000]

# Dummy metrics (users 0-4 stagnate, users 5-9 improve a lot)
metrics = [{'train_loss': 0.1} for _ in range(5)] + [{'train_loss': 0.01} for _ in range(5)]

# Seed the file
k_list = [500] * 10
with open('K_alpha=10_gamma=10_test=0.pickle', 'wb') as f:
    pickle.dump(k_list, f)
    
if os.path.exists('adafl_k_tracker.pickle'):
    os.remove('adafl_k_tracker.pickle')

print("=== Round 1 (Initial Call) ===")
layerSparsification(weights, history, oldseperation, metrics)

print("\n=== Round 2 (Adapting) ===")
history.round = 3
# Provide new metrics to trigger delta L
metrics2 = [{'train_loss': 0.09} for _ in range(5)] + [{'train_loss': 0.001} for _ in range(5)]
layerSparsification(weights, history, oldseperation, metrics2)

print("\n=== Round 3 (More Adapting) ===")
history.round = 4
metrics3 = [{'train_loss': 0.089} for _ in range(5)] + [{'train_loss': 0.0001} for _ in range(5)]
layerSparsification(weights, history, oldseperation, metrics3)
