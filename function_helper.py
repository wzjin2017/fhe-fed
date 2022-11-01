from unittest import result
import torch
import torch
import numpy as np
from collections import OrderedDict

def tensorToNumpyArr(params_tensor):
    params_np = OrderedDict()
    #params_shape = OrderedDict()
    for key in params_tensor.keys():
        params_np[key] = torch.flatten(params_tensor[key]).numpy()
    return params_np

def numpyArrToTensor(params_np, params_shape):
    params_tensor = OrderedDict()
    for key in params_np.keys():
        params_tensor[key] = torch.from_numpy(params_np[key])
        #needs torch.Size() to tuple
        params_tensor[key] = torch.reshape(params_tensor[key], tuple(list((params_shape[key]))))
    return params_tensor

def tensorShape(params_tensor):
    params_shape = OrderedDict()
    for key in params_tensor.keys():
        params_shape[key] = params_tensor[key].size()
    return params_shape
