# -*- coding: utf-8 -*-
"""DSFL aggregation: flatten, Random_Selection, deflatten. Run from repo root with PYTHONPATH=.:common."""
from common.flatten import Flatten, DeFlatten
from methods.dsfl.random_selection import Random_Selection


def alastor(Weights, history):
    flatten_weighted_weights = [Flatten(user)[0] for user in Weights[:]]
    map_shape = Flatten(Weights[0])[1]
    new_flatten_weighted_weights = Random_Selection(flatten_weighted_weights[:], history)
    output = [DeFlatten(user, map_shape) for user in new_flatten_weighted_weights]
    return output
