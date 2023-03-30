import tenseal as ts
import torch
import pickle
import os
from sys import getsizeof

def context():
	context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
	context.global_scale = pow(2, 52)
	context.generate_galois_keys()
	return context

context = context()

plain1 = ts.plain_tensor(torch.rand(1663370))
encrypted_tensor1 = ts.ckks_vector(context, plain1)

with open('ciphertext_seal.pickle', 'wb') as handle:
    pickle.dump(encrypted_tensor1.serialize(), handle, protocol=pickle.HIGHEST_PROTOCOL)
cipher_size = os.path.getsize('ciphertext_seal.pickle')
print(cipher_size)
print(getsizeof(encrypted_tensor1.serialize()))