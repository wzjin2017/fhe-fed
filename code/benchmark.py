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

# Logistic regression model

input_size = 100
num_classes = 1

model_lr = nn.Linear(input_size, num_classes)

#TimeSeriesTransformer

# Initializing a default Time Series Transformer configuration
configuration = TimeSeriesTransformerConfig()

# Randomly initializing a model (with random weights) from the configuration
model_tst = TimeSeriesTransformerModel(configuration)

# MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
model_mlp= MLP()


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 100)
        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out
model_lenet = LeNet()

# RNN(2 LSTM + 1 FC)
class RNN_OriginalFedAvg(nn.Module):
    """Creates a RNN model using LSTM layers for Shakespeare language models (next character prediction task).
    This replicates the model structure in the paper:
    Communication-Efficient Learning of Deep Networks from Decentralized Data
      H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agueray Arcas. AISTATS 2017.
      https://arxiv.org/abs/1602.05629
    This is also recommended model by "Adaptive Federated Optimization. ICML 2020" (https://arxiv.org/pdf/2003.00295.pdf)
    Args:
      vocab_size: the size of the vocabulary, used as a dimension in the input embedding.
      sequence_length: the length of input sequences.
    Returns:
      An uncompiled `torch.nn.Module`.
    """

    def __init__(self, embedding_dim=8, vocab_size=90, hidden_size=256):
        super(RNN_OriginalFedAvg, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)
        # Note that the order of mini-batch is random so there is no hidden relationship among batches.
        # So we do not input the previous batch's hidden state,
        # leaving the first hidden state zero `self.lstm(embeds, None)`.
        lstm_out, _ = self.lstm(embeds)
        # use the final hidden state as the next character prediction
        final_hidden_state = lstm_out[:, -1]
        output = self.fc(final_hidden_state)
        # For fed_shakespeare
        # output = self.fc(lstm_out[:,:])
        # output = torch.transpose(output, 1, 2)
        return output
    
model_rnn= RNN_OriginalFedAvg()

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

"""mobilenet in pytorch
[1] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam

    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861
"""
import logging

class DepthSeperabelConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size,
                groups=input_channels,
                **kwargs
            ),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class MobileNet(nn.Module):
    """
    Args:
        width multipler: The role of the width multiplier α is to thin
                         a network uniformly at each layer. For a given
                         layer and width multiplier α, the number of
                         input channels M becomes αM and the number of
                         output channels N becomes αN.
    """

    def __init__(self, width_multiplier=1, class_num=100):
        super().__init__()

        alpha = width_multiplier
        self.stem = nn.Sequential(
            BasicConv2d(3, int(32 * alpha), 3, padding=1, bias=False),
            DepthSeperabelConv2d(
                int(32 * alpha), int(64 * alpha), 3, padding=1, bias=False
            ),
        )

        # downsample
        self.conv1 = nn.Sequential(
            DepthSeperabelConv2d(
                int(64 * alpha), int(128 * alpha), 3, stride=2, padding=1, bias=False
            ),
            DepthSeperabelConv2d(
                int(128 * alpha), int(128 * alpha), 3, padding=1, bias=False
            ),
        )

        # downsample
        self.conv2 = nn.Sequential(
            DepthSeperabelConv2d(
                int(128 * alpha), int(256 * alpha), 3, stride=2, padding=1, bias=False
            ),
            DepthSeperabelConv2d(
                int(256 * alpha), int(256 * alpha), 3, padding=1, bias=False
            ),
        )

        # downsample
        self.conv3 = nn.Sequential(
            DepthSeperabelConv2d(
                int(256 * alpha), int(512 * alpha), 3, stride=2, padding=1, bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha), int(512 * alpha), 3, padding=1, bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha), int(512 * alpha), 3, padding=1, bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha), int(512 * alpha), 3, padding=1, bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha), int(512 * alpha), 3, padding=1, bias=False
            ),
            DepthSeperabelConv2d(
                int(512 * alpha), int(512 * alpha), 3, padding=1, bias=False
            ),
        )

        # downsample
        self.conv4 = nn.Sequential(
            DepthSeperabelConv2d(
                int(512 * alpha), int(1024 * alpha), 3, stride=2, padding=1, bias=False
            ),
            DepthSeperabelConv2d(
                int(1024 * alpha), int(1024 * alpha), 3, padding=1, bias=False
            ),
        )

        self.fc = nn.Linear(int(1024 * alpha), class_num)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def mobilenet(alpha=1, class_num=100):
    logging.info("class_num = " + str(class_num))
    return MobileNet(alpha, class_num)
model_mobile= MobileNet()

#Resnet-18
class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
# model_res18 = ResNet_18(1, 10)

from torchvision import datasets, models, transforms
model_res18 = models.resnet18(pretrained=True)
#resnet-34
model_res34 = models.resnet34(pretrained=True)
#resnet-50
model_res50 = models.resnet50(pretrained=True)
#GroupViT
from transformers import GroupViTConfig, GroupViTModel
configuration1 = GroupViTConfig()
model_group = GroupViTModel(configuration1)

#ViT
from transformers import ViTConfig, ViTModel
configuration = ViTConfig()
model_vit = ViTModel(configuration)

#BERT
from transformers import BertConfig, BertModel
# Initializing a BERT bert-base-uncased style configuration
configuration_bert = BertConfig()

# Initializing a model (with random weights) from the bert-base-uncased style configuration
model_bert = BertModel(configuration_bert)
###########################################################

## Benchmark params
#update number of clients
n_clients = 3
n_times = 1
#update models
model = model_res18
# with open('model.pickle', 'wb') as handle:
# 	pickle.dump(model.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
#summary(model_lr, (1, 100))



t_plain_list = []
t_cipher_list = []
t_init_list = []
t_enc_list = []
t_agg_list = []
t_dec_list = []
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
	plain_aggregate(global_model, client_models)
	time_plain_e = time.time()
	time_plain = time_plain_e - time_plain_s
	t_plain += time_plain
	#print("Plaintext aggregation done.\n")
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
print("Plaintext Time: {}".format(t_plain))
print("Init Time: {}".format(t_init))
print("Encryption Time: {}".format(t_enc))
print("Secure Agg Time: {}".format(t_agg))
print("Decryption Time: {}".format(t_dec))
t_cipher = t_init + t_enc + t_agg + t_dec
t_plain_list.append(t_plain)
t_cipher_list.append(t_cipher)
t_init_list.append(t_init)
t_enc_list.append(t_enc)
t_agg_list.append(t_agg)
t_dec_list.append(t_dec)
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


