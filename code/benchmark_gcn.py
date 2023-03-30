import SHELFI_FHE as m
import torch.nn as nn
from torchinfo import summary
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
from torch_geometric.nn import GCNConv

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

def plain_aggregate(global_model, client_models):
	global_dict = global_model.state_dict()
	for k in global_dict.keys():
		for i in range(1, len(client_models)):
			global_dict[k] += client_models[i].state_dict()[k]
		global_dict[k] = torch.div(global_dict[k], len(client_models))
		global_model.load_state_dict(global_dict)
	for model in client_models:
		model.load_state_dict(global_model.state_dict())
## Models

class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, NumLayers):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(nfeat, nhid, normalize=True, cached=True))
        for _ in range(NumLayers - 2):
            self.convs.append(
                GCNConv(nhid, nhid, normalize=True, cached=True))
        self.convs.append(
            GCNConv(nhid, nclass, normalize=True, cached=True))
        self.dropout = dropout
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)
model_gcn = GCN(1433, 16, 0.5, 2)
###########################################################

## Benchmark params
#update number of clients
n_clients = 5
n_times = 1
#update models
model = model_gcn
# with open('model.pickle', 'wb') as handle:
# 	pickle.dump(model.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
#summary(model_lr, (1, 100))


t_plain_list = []
t_cipher_list = []
N =5
t_plain = 0.0
t_init = 0.0
t_enc = 0.0
t_agg = 0.0
t_dec = 0.0
global_model = copy.deepcopy(model)
client_models = [copy.deepcopy(global_model) for i in range(N)]
for i_try in range(n_times):
	time_plain_s = time.time()
	plain_aggregate(global_model, client_models)
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

	#print("Setup CryptoContext.")
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
		for id in range(N):
			enc_learner_layer.append(OrderedDict())
			enc_learner_layer[id][key] = FHE_helper.encrypt(learner_data_layer[id][key])
	time_enc_e = time.time()
	#print("Encrytion done.\n")

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

	t_plain = t_plain / n_times
	t_init = t_init / n_times
	t_enc = t_enc / n_times
	t_agg = t_agg / n_times
	t_dec = t_dec / n_times
	# print("Plaintext Time: {}".format(t_plain))
	# print("Init Time: {}".format(t_init))
	# print("Encryption Time: {}".format(t_enc))
	# print("Secure Agg Time: {}".format(t_agg))
	# print("Decryption Time: {}".format(t_dec))
	t_cipher = t_init + t_enc + t_agg + t_dec
	t_plain_list.append(t_plain)
	t_cipher_list.append(t_cipher)
	del learner_data_layer
	del enc_learner_layer
	del eval_data
	del final_data
print(t_cipher_list)
print(t_plain_list)

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


