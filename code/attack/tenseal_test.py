import torch.nn as nn
import numpy as np
import time
import torch
import copy
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import tenseal as ts
import pickle


context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.global_scale = pow(2, 52)
context.generate_galois_keys()


with open('context.pickle', 'wb') as handle:
    pickle.dump(context.serialize(), handle, protocol=pickle.HIGHEST_PROTOCOL)
