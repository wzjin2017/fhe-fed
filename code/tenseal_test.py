import numpy as np
import time
import copy
import tenseal as ts
import pickle
import torch

context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.global_scale = pow(2, 52)
context.generate_galois_keys()

# plaintext = ts.plain_tensor(torch.flatten(torch.ones([2, 4], dtype=torch.float64)))
# ciphertext  = ts.ckks_vector(context, plaintext)
# decrypted = ciphertext.decrypt()
# print(type(torch.FloatTensor(decrypted)))
with open('context.pickle', 'wb') as handle:
    pickle.dump(context.serialize(save_secret_key=True), handle, protocol=pickle.HIGHEST_PROTOCOL)
