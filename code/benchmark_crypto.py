import SHELFI_FHE as m
import torch.nn as nn
import numpy as np
import time
import torch
import copy
import torch.nn.functional as F
from collections import OrderedDict
import pickle
import matplotlib.pyplot as plt
import sys
import csv
import os
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

### Benchmark for different crypo parameter setups

# Load test dataset
transform = transforms.Compose([transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])

test_data = FashionMNIST(
    root = 'data', 
	download=True,
    train = False, 
    transform = transform
)
loader = torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1)

# Some helper functions
def test(model, loader):
    # Test the model
	correct = 0.0
	model.eval()
	for (i, (inputs, labels)) in enumerate(loader):
		inputs = torch.squeeze(inputs, 1)
		scores = model(inputs)
		# Count how many correct in this batch.
		max_scores, max_labels = scores.max(1)
		correct += (max_labels == labels).sum().item()
	accuracy = correct / len(test_data)
	return accuracy

def tensor_to_numpy_arr(params_tensor):
    params_np = OrderedDict()
    #params_shape = OrderedDict()
    for key in params_tensor.keys():
        params_np[key] = torch.flatten(params_tensor[key]).numpy()
    return params_np

def numpy_arr_to_tensor(params_np, params_shape):
    params_tensor = OrderedDict()
    for key in params_np.keys():
        params_tensor[key] = torch.from_numpy(params_np[key])
        #needs torch.Size() to tuple
        params_tensor[key] = torch.reshape(params_tensor[key], tuple(list((params_shape[key]))))
    return params_tensor

def tensor_shape(params_tensor):
    params_shape = OrderedDict()
    for key in params_tensor.keys():
        params_shape[key] = params_tensor[key].size()
    return params_shape

def plain_aggregate(global_model, client_models):
	global_dict = global_model.state_dict()
	for k in global_dict.keys():
		for i in range(1, len(client_models)):
			global_dict[k] += client_models[i].state_dict()[k]
		global_dict[k] = torch.div(global_dict[k], len(client_models))
		global_model.load_state_dict(global_dict)
	for model in client_models:
		model.load_state_dict(global_model.state_dict())
###########################################################
## Models

#CNN
class CNN_OriginalFedAvg(torch.nn.Module):

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

###########################################################

## Benchmark params
#update number of clients
N = 3
n_times = 1
model = model_cnn


batch_size_list = [1024, 2048, 4096]
scaling_bit_list = [14, 20, 33, 40, 52]
# batch_size_list = [2048, 4096]
# scaling_bit_list = [14, 20]
configs = []
for a in  batch_size_list:
	for b in scaling_bit_list:
		configs.append((a, b))

result = []


for (batch_size, scaling_bit) in configs:
	print("Batch: {} and scaling: {}".format(batch_size,scaling_bit))
	t_plain = 0.0
	t_init = 0.0
	t_enc = 0.0
	t_agg = 0.0
	t_dec = 0.0

	params_shape = OrderedDict()
	for key in model.state_dict().keys():
		params_shape[key] = model.state_dict()[key].size()


	global_model = copy.deepcopy(model)
	client_models = [copy.deepcopy(global_model) for i in range(N)]
	fhe_global_model = copy.deepcopy(model)
	for i_try in range(n_times):
		plain_aggregate(global_model, client_models)
		#print("Plaintext aggregation done.\n")
		del client_models
		acc_plain = test(global_model, loader)
		del global_model
		# FHE
		learner_data_layer = []
		params = tensor_to_numpy_arr(model.state_dict())
		learner_data_layer = []
		params = tensor_to_numpy_arr(model.state_dict())

		for id in range(N):
			learner_data_layer.append(params)

		scalingFactors = np.full(N, 1/N).tolist()
		time_init_s = time.time()

		#print("Setup CryptoContext.")
		FHE_helper = m.CKKS("ckks", batch_size, scaling_bit, "./resources/cryptoparams/")
		#FHE_helper = m.CKKS()

		FHE_helper.genCryptoContextAndKeyGen()
		FHE_helper.loadCryptoParams()
		time_init_e = time.time()
		time_init = time_init_e - time_init_s
		t_init += time_init

		#encrypting
		enc_learner_layer = []

		time_enc_s = time.time()
		for key in learner_data_layer[0].keys():
			for id in range(N):
				enc_learner_layer.append(OrderedDict())
				enc_learner_layer[id][key] = FHE_helper.encrypt(learner_data_layer[id][key])
		time_enc_e = time.time()
		#print("Encrytion done.\n")
		with open('ciphertext_cnn.pickle', 'wb') as handle:
			pickle.dump(enc_learner_layer[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
		cipher_size = os.path.getsize('ciphertext_cnn.pickle')
		time_enc = (time_enc_e - time_enc_s)/N
		t_enc += time_enc
		
		#print(FHE_helper.decrypt(enc_res_learner[0][“conv1.weight”], int(learner_data_layer[0][“fc1.weight”].size)))


		#weighted average
		eval_data = copy.deepcopy(learner_data_layer[0])

		time_agg_s = time.time()
		for key in enc_learner_layer[0].keys():
			leaner_layer_temp = []
			for id in range(N):
				leaner_layer_temp.append(enc_learner_layer[id][key])
				#print(leaner_layer_temp)
			#print(key)
			eval_data[key] = FHE_helper.computeWeightedAverage(leaner_layer_temp, scalingFactors)
		time_agg_e = time.time()
		
		#print("Secure FedAvg done.\n")
		time_agg = (time_agg_e - time_agg_s)
		t_agg += time_agg

		#decryption
		model_size = OrderedDict()
		for key in model.state_dict().keys():
			model_size[key] = torch.flatten(model.state_dict()[key]).numpy().size
		final_data = OrderedDict()

		time_dec_s = time.time()
		for key in learner_data_layer[0].keys():
			final_data[key] = FHE_helper.decrypt(eval_data[key], model_size[key])
		time_dec_e = time.time()
		#print("Decryption done.\n")
		time_dec = (time_dec_e - time_dec_s)
		t_dec += time_dec
		dec_model = OrderedDict() 
		for key in learner_data_layer[0].keys():
			dec_model[key] = torch.from_numpy(final_data[key])
            # need torch.Size() to tuple
			dec_model[key] = torch.reshape(dec_model[key], tuple(list((params_shape[key]))))
		fhe_global_model.load_state_dict(dec_model)

	t_plain = t_plain / n_times
	t_init = t_init / n_times
	t_enc = t_enc / n_times
	t_agg = t_agg / n_times
	t_dec = t_dec / n_times
	t_cipher = t_init + t_enc + t_agg + t_dec
	# print("Plaintext Time: {}".format(t_plain))
	# print("Init Time: {}".format(t_init))
	# print("Encryption Time: {}".format(t_enc))
	# print("Secure Agg Time: {}".format(t_agg))
	# print("Decryption Time: {}".format(t_dec))
	acc_fhe = test(fhe_global_model, loader)
	print(acc_plain)
	print(acc_fhe)
	acc_delta =  acc_plain - acc_fhe
	result.append([batch_size, scaling_bit, t_cipher, cipher_size, acc_delta])

	del learner_data_layer
	del enc_learner_layer
	del eval_data
	del final_data

#print(result)

fields = ['Batch Size', 'Scaling Factor Bits', 'Computation', 'Communication', 'Acc Delta']

with open('params_results.csv', 'w') as f:
	write = csv.writer(f)
	write.writerow(fields)
	for i in result:
		write.writerow(i)



# with open('plain_number.pickle', 'wb') as handle:
# 	pickle.dump(t_plain_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('fhe_number.pickle', 'wb') as handle:
# 	pickle.dump(t_cipher_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

# sns.set_style("whitegrid")
# number = [i for i in range(2, n_clients, 3)]
# #plt.plot(number, t_plain_list, color='tab:blue',linewidth=2,label='Plaintext',linestyle='-')
# plt.plot(number, t_cipher_list, color='tab:red',linewidth=2, label='FHE',linestyle='-')

# plt.xlabel("Number of Clients")
# plt.ylabel("Execution Time (s)")
# plt.legend(loc = 'best')
# plt.savefig('client_number.pdf', bbox_inches='tight')


