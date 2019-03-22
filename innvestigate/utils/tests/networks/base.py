from __future__ import\
        absolute_import, print_function, division, unicode_literals
from builtins import range

import torch
import torch.nn as nn

__all__ = [
    "log_reg", 

    "mlp_2dense",
    "mlp_3dense",

    "cnn_1convb_2dense",
    "cnn_2convb_2dense",
    "cnn_2convb_3dense",
    "cnn_3convb_3dense",
]

# TODO: more consistent naming

