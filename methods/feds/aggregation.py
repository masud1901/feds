# -*- coding: utf-8 -*-
"""FEDS aggregation: flatten, adaptive layer sparsification, deflatten. Run from repo root with PYTHONPATH=.:common."""
from common.flatten import Flatten, DeFlatten
from methods.feds.sparsification import layerSparsification


def alastor_feds(Weights, history, metrics=None):
    flatten_weighted_weights = [Flatten(user)[0] for user in Weights[:]]
    map_shape = Flatten(Weights[0])[1]
    new_flatten_weighted_weights = layerSparsification(
        flatten_weighted_weights[:], history, map_shape["Seperation"], metrics
    )
    output = [DeFlatten(user, map_shape) for user in new_flatten_weighted_weights]
    return output
