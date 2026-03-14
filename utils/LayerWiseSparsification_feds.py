# -*- coding: utf-8 -*-
"""
Created on Fri May  6 23:03:34 2022

@author: Mahdi
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from FindingKSpeech import findK


def layerSparsification(flatten_weights, history, oldseperation, metrics=None):
    """ Input: flattern_weighted_weights--> List __> with size of number of users
                In each index of this list is a flatten numpy array with size for instane M
                                            
    
        Output: --> Should be a list with size of number of users
                    and each index of this list should be a numpy array with size M            
    """
    # Set parameters
    iteration = history.round - 1 # global round
    num_user = len(flatten_weights)
    dynamic_sparsification = True  # If false we use the minimum of K
    
    # FEDS Adaptive K logic
    # Initialize or load K_i
    try:
        with open('feds_k_tracker.pickle', 'rb') as f:
            k_tracker = pickle.load(f)
            k_list = k_tracker['k_list']
            prev_losses = k_tracker['prev_losses']
    except FileNotFoundError:
        # Initial K (from original DSFL file or uniform initialization)
        with open('K_alpha=10_gamma=10_test=0.pickle', 'rb') as f:
            k_list = pickle.load(f)
        prev_losses = [None] * num_user
        
    seperation = oldseperation[::2]
    # print(seperation)
    
    # Calculate total d (maximum parameters)
    d = len(flatten_weights[0])
    K_min = 100 # arbitrary min to allow at least some learning
    K_max = d
    eta = 5000000.0  # learning rate for K. high because loss diff is tiny.
                     # e.g., diff = 0.01 -> K shift = 50000 parameters.
    
    if iteration >= 1 and metrics is not None:
        # Calculate Delta L and tau
        delta_L = []
        for i in range(num_user):
            current_loss = metrics[i].get('train_loss', 0)
            if prev_losses[i] is not None:
                delta_L.append(prev_losses[i] - current_loss)
            else:
                delta_L.append(0)
            prev_losses[i] = current_loss

        # Tau is the average loss improvement in this round
        tau = np.mean(delta_L) if len(delta_L) > 0 else 0
        
        # FEDS Update Rule
        for i in range(num_user):
            if type(k_list[i]) is list:
                 k_list[i] = sum(k_list[i]) # Flatten to scalar if needed
            k_list[i] = int(np.clip(k_list[i] - eta * (delta_L[i] - tau), K_min, K_max))
            
    # Save tracker
    with open('feds_k_tracker.pickle', 'wb') as f:
        pickle.dump({'k_list': k_list, 'prev_losses': prev_losses}, f)
            
    print("FEDS Adaptive K values:", k_list)
    
    # If CKA is needed
    if iteration > 1:
        cka = findK()
    # print(cka)
    
    # Testing
    # for idx, user in enumerate(range(num_user)):
    #     print("actual", idx, flatten_weights[user][:10])
        


    # Calculate model difference
    model_difference = [user_weights - history.globals[iteration]
                        for user_weights in flatten_weights]

    # Calculate the error accumulated models
    model_difference_AccError = [model_difference[i] + history.error[iteration][i] for i in range(num_user)]

    # Calculate the mask list
    # mask_list = [np.array([0]*len(flatten_weights[0])) for _ in range(num_user)]
    # for layer in range(len(seperation)-1):
    #     for user in range(num_user):
    #         if k_list[user] != 0:
    #             layer_model = np.absolute(model_difference_AccError[user][seperation[layer]:seperation[layer+1]])
    #             for index in np.argpartition(layer_model, -k_list[user][layer])[-k_list[user][layer]:].tolist():
    #                 mask_list[user][seperation[layer]+index] = 1
    
    # """New method"""
    # mult_list = [np.array([0.0]*len(model_difference_AccError[0])) for _ in range(num_user)]
    # for user in range(num_user):
    #     for layer in range(len(seperation)-1):
    #         for index in range (seperation[layer+1]-seperation[layer]):
    #             mult_list[user][seperation[layer] + index] = cka[user][layer]
                
    """New New method"""
    mult_list = [np.array([1.0]*len(model_difference_AccError[0])) for _ in range(num_user)]
    if iteration > 2:
        for user in range(num_user):
            for layer in range(len(seperation)-1):
                for index in range (seperation[layer+1]-seperation[layer]):
                    mult_list[user][seperation[layer] + index] = np.random.binomial(1, 1-cka[user][layer])


    # Calculate the mask list
    mask_list = [np.array([0]*len(model_difference_AccError[0])) for _ in range(num_user)]
    for user in range(num_user):
        if k_list[user] != 0:
            mult = np.multiply(model_difference_AccError[user], mult_list[user])
            for index in np.argpartition(np.absolute(mult), -k_list[user])[-k_list[user]:].tolist():
                mask_list[user][index] = 1
                
    # """New Very new method"""
    # mult_list = [np.array([0]*len(model_difference_AccError[0])) for _ in range(num_user)]
    # for user in range(num_user):
    #     for layer in range(len(seperation)-1):
    #         for index in range (seperation[layer+1]-seperation[layer]):
    #             mult_list[user][seperation[layer] + index] = np.random.binomial(1, 1)


    # # Calculate the mask list
    # mask_list = [np.array([0]*len(model_difference_AccError[0])) for _ in range(num_user)]
    # for user in range(num_user):
    #     if k_list[user] != 0:
    #         for index in np.argpartition(np.absolute(model_difference_AccError[user]), -k_list[user])[-k_list[user]:].tolist():
    #             mask_list[user][index] = 1
    #     mask_list[user] = np.multiply(mask_list[user], mult_list[user])
    
    
    print([sum(mask_list[i]) for i in range(num_user)])
    # Calculate the model for sending
    model_to_send = [np.multiply(mask_list[user], model_difference_AccError[user]) for user in range(num_user)]
    
    # Plot
    # for layer in range(len(seperation)-1):
    #     plt.figure()
    #     if layer == 4:
    #         plt.figure(figsize = (50, 5))
    #     plt.bar([i for i in range(seperation[layer+1] - seperation[layer])] , model_to_send[0][seperation[layer]:seperation[layer+1]])
    #     plt.savefig(f"layerwisefigs\{iteration}{layer}.png")
    
    # Update the error
    new_error = [np.multiply(np.array([1]*len(mask_list[0]))- mask_list[user], model_difference_AccError[user]) for user in range(num_user)]
    history.updateError(new_error)

    # Average the model differences
    global_model_difference = model_to_send[0]
    for user in range(1, num_user):
        global_model_difference += model_to_send[user]
    global_model_difference /= len(flatten_weights)

    # Calculate the next global round
    global_model =  history.globals[iteration] + global_model_difference
    history.updateGlobal(global_model)

    # Dummy global model
    global_model = [global_model for _ in range(num_user)]

    return global_model 
