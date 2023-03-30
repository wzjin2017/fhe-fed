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
import pickle
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel
import os
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

###########################################################

## Benchmark params
#update number of clients
n_clients = 1
n_times = 1
#update models
# with open('model.pickle', 'wb') as handle:
# 	pickle.dump(model.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
# pytorch_total_params = sum(p.numel() for p in model.parameters())
# print(pytorch_total_params)
# #summary(model_lr, (1, 100))

# def context():
# 	context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
# 	context.global_scale = pow(2, 52)
# 	context.generate_galois_keys()
# 	return context

models = [model_lr, model_tst, model_mlp, model_lenet, model_rnn, model_cnn, model_mobile, model_res18, model_res34, model_res50, model_group, model_vit, model_bert]



N = n_clients
model_counter = 0
for model in models:
	for i_try in range(n_times):
		t_init = 0.0
		t_enc = 0.0
		t_agg = 0.0
		t_dec = 0.0
		plaintext_list = []
		params = model.state_dict()

		scalingFactors = copy.deepcopy(params)
		for id in range(N):
			plaintext_list.append(copy.deepcopy(params))
		time_init_s = time.time()
		# init tenseal context
		context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
		context.global_scale = pow(2, 52)
		context.generate_galois_keys()
                
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
			scalingFactors[key] = ts.plain_tensor(torch.flatten(torch.full_like(scalingFactors[key], 0.3)))

		for id in range(N):
			for key in plaintext_list[0].keys():
				ciphertext_list[id][key] = ts.ckks_vector(context, plaintext_list[id][key]).serialize()
		# print("Encrytion done.\n")
		with open('ciphertext_nvidia.pickle', 'wb') as handle:
			pickle.dump(ciphertext_list[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
		cipher_size = os.path.getsize('ciphertext_nvidia.pickle')
		print(cipher_size)
		del plaintext_list
		del ciphertext_list
		
	with open('nvidia_results_file.txt', 'a') as f:
		f.write("Model #"+str(model_counter)+"\n")
		f.write("File Size: {}".format(cipher_size)+"\n")
	model_counter += 1
    

