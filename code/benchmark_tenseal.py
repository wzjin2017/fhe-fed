import SHELFI_FHE as m
import torch.nn as nn
import numpy as np
import time
import torch
import copy
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import tenseal as ts
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel


#CNN
class CNN_OriginalFedAvg(torch.nn.Module):
    """The CNN model used in the original FedAvg paper:
    "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    https://arxiv.org/abs/1602.05629.

    The number of parameters when `only_digits=True` is (1,663,370), which matches
    what is reported in the paper.
    When `only_digits=True`, the summary of returned model is

    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 32)        832
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    flatten (Flatten)            (None, 3136)              0
    _________________________________________________________________
    dense (Dense)                (None, 512)               1606144
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                5130
    =================================================================
    Total params: 1,663,370
    Trainable params: 1,663,370
    Non-trainable params: 0

    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=True):
        super(CNN_OriginalFedAvg, self).__init__()
        self.only_digits = only_digits
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.linear_2 = nn.Linear(512, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        # x = self.softmax(self.linear_2(x))
        return x
model_cnn= CNN_OriginalFedAvg()

# Logistic regression model

input_size = 100
num_classes = 1

model_lr = nn.Linear(input_size, num_classes)

#TimeSeriesTransformer

# Initializing a default Time Series Transformer configuration
configuration = TimeSeriesTransformerConfig()

# Randomly initializing a model (with random weights) from the configuration
model_tst = TimeSeriesTransformerModel(configuration)

from transformers import BertConfig, BertModel
# Initializing a BERT bert-base-uncased style configuration
configuration_bert = BertConfig()

# Initializing a model (with random weights) from the bert-base-uncased style configuration
model_bert = BertModel(configuration_bert)

from torchvision import datasets, models, transforms
#resnet-50
model_res50 = models.resnet50(pretrained=True)
###########################################################

## Benchmark params
#update number of clients
n_clients = 3
n_times = 1
#update models
model = model_tst
# with open('model.pickle', 'wb') as handle:
# 	pickle.dump(model.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
#summary(model_lr, (1, 100))

def context():
	context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
	context.global_scale = pow(2, 52)
	context.generate_galois_keys()
	return context



N = n_clients
t_init = 0.0
t_enc = 0.0
t_agg = 0.0
t_dec = 0.0
plaintext_list = []
params = model.state_dict()
for i_try in range(n_times):
	
	scalingFactors = copy.deepcopy(params)
	for id in range(N):
		plaintext_list.append(copy.deepcopy(params))


	

	time_init_s = time.time()

	context = context()
	time_init_e = time.time()
	time_init = time_init_e - time_init_s
	t_init += time_init

	#encrypting
	ciphertext_list = copy.deepcopy(plaintext_list)
	scalingFactors = copy.deepcopy(plaintext_list[0])

	time_enc_s = time.time()


	for id in range(N):
		for key in plaintext_list[0].keys():
			plaintext_list[id][key] = ts.plain_tensor(torch.flatten(plaintext_list[id][key]))

	for key in scalingFactors.keys():
		scalingFactors[key] = ts.plain_tensor(torch.flatten(torch.full_like(scalingFactors[key], 0.1)))
	
	print("Done Prep")

	for id in range(N):
		for key in plaintext_list[0].keys():
			ciphertext_list[id][key] = ts.ckks_vector(context, plaintext_list[id][key])
	time_enc_e = time.time()
	print("Encrytion done.\n")
	del plaintext_list
	time_enc = (time_enc_e - time_enc_s)/N
	t_enc += time_enc
	


	#weighted average

	#ciphertext_result = copy.deepcopy(ciphertext_list[0])
	time_agg_s = time.time()
	
	for key in scalingFactors.keys():
		for id in range(N):
			if id != 0:
				temp = ciphertext_list[id][key] * scalingFactors[key]
				ciphertext_list[0][key] = ciphertext_list[0][key] + temp

	time_agg_e = time.time()
	
	print("Secure FedAvg done.\n")
	time_agg = (time_agg_e - time_agg_s)
	t_agg += time_agg

	#decryption
	time_dec_s = time.time()
	plain_result = copy.deepcopy(params)
	for key in scalingFactors.keys():
		plain_result[key] = ciphertext_list[0][key].decrypt()

	time_dec_e = time.time()
	print("Decryption done.\n")
	time_dec = (time_dec_e - time_dec_s)
	t_dec += time_dec

t_init = t_init / n_times
t_enc = t_enc / n_times
t_agg = t_agg / n_times
t_dec = t_dec / n_times

print("Init Time: {}".format(t_init))
print("Encryption Time: {}".format(t_enc))
print("Secure Agg Time: {}".format(t_agg))
print("Decryption Time: {}".format(t_dec))
t_cipher = t_init + t_enc + t_agg + t_dec

