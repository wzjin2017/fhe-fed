import SHELFI_FHE as m
import torch.nn as nn
import numpy as np
import time
import torch
import copy
import torch.nn.functional as F
from collections import OrderedDict
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
import re

# Some helper functions

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

def plain_aggregate(global_model, client_models, num_enc_layer):
	global_dict = global_model.state_dict()
	for k in global_dict.keys():
		new_k = re.sub("[^0-9]", "", k)
		if new_k == '' or int(new_k) not in num_enc_layer:
			for i in range(1, len(client_models)):
				global_dict[k] += client_models[i].state_dict()[k]
			global_dict[k] = torch.div(global_dict[k], len(client_models))

def count_parameters(model):
    table = PrettyTable(["Layer Name", "Parameters Listed"])
    t_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        t_params+=param
    print(table)
    print(f"Sum of trained paramters: {t_params}")
    return table
## Models

from torchvision import datasets, models, transforms
#resnet-50
model_res50 = models.resnet50(pretrained=True)

#ViT
from transformers import ViTConfig, ViTModel
configuration = ViTConfig()
model_vit = ViTModel(configuration)

###########################################################

## Benchmark params
#update number of clients
n_clients = 3
n_times = 1
#update models
model = model_res50
# with open('model.pickle', 'wb') as handle:
# 	pickle.dump(model.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
table = count_parameters(model)
#print(model.state_dict())


# num_enc_layer = [0, 1, 2, 3]
# total_param_enc = 0
# for name, parameter in model.named_parameters():
# 	new_k = re.sub("[^0-9]", "", name)
# 	if new_k != '':
# 		if int(new_k) in num_enc_layer:
# 			total_param_enc += parameter.numel()
# print(total_param_enc/86389248)

'''


N = n_clients

t_plain = 0.0
t_init = 0.0
t_enc = 0.0
t_agg = 0.0
t_dec = 0.0
global_model = copy.deepcopy(model)
client_models = [copy.deepcopy(global_model) for i in range(N)]

for i_try in range(n_times):
	time_plain_s = time.time()
	plain_aggregate(global_model, client_models, num_enc_layer)
	time_plain_e = time.time()
	time_plain = time_plain_e - time_plain_s
	t_plain += time_plain
	print("Plaintext aggregation done.\n")
	del global_model
	del client_models
	learner_data_layer = []
	params = tensor_to_numpy_arr(model.state_dict())

	for id in range(N):
		learner_data_layer.append(params)

	# loading model params from files
	# for id in range(3):
	#     os.system(“python3 tabcnn_learner.py ” + str(id+1)+ ” ” + str(n)) 
	#     with open(“models/model”+str(id+1)+“.pickle”, ‘rb’) as handle:
	#         b = pickle.load(handle)
	#     learner_data_layer.append(b)
	#     with open(“models/tensor_model”+str(id+1)+“.pickle”, ‘rb’) as handle:
	#         c = pickle.load(handle)
	#     plaintext_data_layer.append(c)


	scalingFactors = np.full(N, 1/N).tolist()
	time_init_s = time.time()

	print("Setup CryptoContext.")
	FHE_helper = m.CKKS("ckks", 4096, 52, "./resources/cryptoparams/")
	#FHE_helper = m.CKKS()

	#FHE_helper.genCryptoContextAndKeyGen()
	FHE_helper.loadCryptoParams()
	time_init_e = time.time()
	time_init = time_init_e - time_init_s
	t_init += time_init

	#encrypting
	enc_learner_layer = []

	time_enc_s = time.time()
	for key in learner_data_layer[0].keys():
		new_k = re.sub("[^0-9]", "", key)
		if new_k != '':
			if int(new_k) in num_enc_layer:
				for id in range(N):
					enc_learner_layer.append(OrderedDict())
					enc_learner_layer[id][key] = FHE_helper.encrypt(learner_data_layer[id][key])
	time_enc_e = time.time()
	print("Encrytion done.\n")

	time_enc = (time_enc_e - time_enc_s)/N
	t_enc += time_enc
	
	#print(FHE_helper.decrypt(enc_res_learner[0][“conv1.weight”], int(learner_data_layer[0][“fc1.weight”].size)))


	#weighted average
	eval_data = copy.deepcopy(enc_learner_layer[0])

	time_agg_s = time.time()
	for key in enc_learner_layer[0].keys():
		leaner_layer_temp = []
		for id in range(N):
			leaner_layer_temp.append(enc_learner_layer[id][key])
			#print(leaner_layer_temp)
		#print(key)
		eval_data[key] = FHE_helper.computeWeightedAverage(leaner_layer_temp, scalingFactors)
	time_agg_e = time.time()
	
	print("Secure FedAvg done.\n")
	time_agg = (time_agg_e - time_agg_s)
	t_agg += time_agg

	#decryption
	model_size = OrderedDict()
	for key in model.state_dict().keys():
		model_size[key] = torch.flatten(model.state_dict()[key]).numpy().size
	final_data = OrderedDict()

	time_dec_s = time.time()
	for key in enc_learner_layer[0].keys():
		final_data[key] = FHE_helper.decrypt(eval_data[key], model_size[key])
	time_dec_e = time.time()
	print("Decryption done.\n")
	time_dec = (time_dec_e - time_dec_s)
	t_dec += time_dec
    
t_plain = t_plain / n_times
t_init = t_init / n_times
t_enc = t_enc / n_times
t_agg = t_agg / n_times
t_dec = t_dec / n_times
print("Plaintext Time: {}".format(t_plain))
print("Init Time: {}".format(t_init))
print("Encryption Time: {}".format(t_enc))
print("Secure Agg Time: {}".format(t_agg))
print("Decryption Time: {}".format(t_dec))
t_total = t_plain+t_init + t_enc + t_agg + t_dec

del learner_data_layer
del enc_learner_layer
del eval_data
del final_data


# with open('plain_number.pickle', 'wb') as handle:
# 	pickle.dump(t_plain_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('fhe_number.pickle', 'wb') as handle:
# 	pickle.dump(t_cipher_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('init_number.pickle', 'wb') as handle:
# 	pickle.dump(t_init_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('enc_number.pickle', 'wb') as handle:
# 	pickle.dump(t_enc_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('agg_number.pickle', 'wb') as handle:
# 	pickle.dump(t_agg_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('dec_number.pickle', 'wb') as handle:
# 	pickle.dump(t_dec_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

# sns.set_style("whitegrid")
# number = [i for i in range(2, n_clients, 3)]
# #plt.plot(number, t_plain_list, color='tab:blue',linewidth=2,label='Plaintext',linestyle='-')
# plt.plot(number, t_cipher_list, color='tab:red',linewidth=2, label='Total',linestyle='-')
# plt.plot(number, t_init_list, color='tab:orange',linewidth=2, label='Init',linestyle='-')
# plt.plot(number, t_enc_list, color='tab:olive',linewidth=2, label='Enc',linestyle='-')
# plt.plot(number, t_agg_list, color='tab:green',linewidth=2, label='Secure Agg',linestyle='-')
# plt.plot(number, t_dec_list, color='tab:gray',linewidth=2, label='Dec',linestyle='-')

# plt.xlabel("Number of Clients")
# plt.ylabel("Execution Time (s)")
# plt.legend(loc = 'best')
# plt.savefig('client_number.pdf', bbox_inches='tight')
'''