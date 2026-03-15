# -*- coding: utf-8 -*-
"""
Stub for Random_Selection used by DSFL aggregation.
Original DSFL uses this for client selection / K sampling.
If you have the original DSFL codebase, replace this with the real Random_Selection implementation.
"""
import numpy as np


def Random_Selection(flatten_weights, history):
    """
    Stub: returns weights unchanged.
    Full DSFL behavior requires the original Random_Selection from the DSFL repo.
    """
    return flatten_weights[:]
