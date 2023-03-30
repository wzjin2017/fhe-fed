import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# with open('fhe_number.pickle', 'rb') as handle:
# 	t_cipher_list = pickle.load(handle)

sns.set_style("whitegrid")
n_clients = 201
number = [101,
5609,
79510,
88648,
822570,
1663370,
3315428,
12556426,
21797672,
25557032,
55726609,
86389248,
109482240]

models= ['Linear',
'TST',
'MLP',
'LeNet',
'RNN',
'CNN',
'MobileNet'
'ResNet-18'
'ResNet-34'
'ResNet-50'
'GViT',
'ViT',
'BERT']


fhe_file = [272384,
544768,
5447680,
6259998,
55021568,
110860288,
220631040,
835401728,
1449900032,
1699948544,
3706056704,
5745123328,
7280824320]
plain_file = [1131,
53912,
319464,
357908,
3293785,
6656175,
13408318,
50309716,
87328827,
102543033,
223166212,
345633211,
438009167]

nvidia_file = [668949,
3008971,
7687223,
9361226,
68858960,
137722599,
321921449,
987154434,
1838229588,
2170521131,
4643220301,
7082031495,
8966672437]


plt.plot(number, fhe_file, color='tab:blue',linewidth=2,marker='s', markersize= 5,markevery=np.where(np.array(number) > 0, True, False),label='Ours',linestyle='-')
plt.plot(number, nvidia_file, color='tab:red',linewidth=2, marker='P', markersize= 5,markevery=np.where(np.array(number) > 0, True, False),label='FLARE',linestyle='-')
plt.plot(number, plain_file, color='tab:olive',linewidth=2, marker='o', markersize= 5,markevery=np.where(np.array(number) > 0, True, False),label='Plaintext',linestyle='-')
# for i, txt in enumerate(models):
#     plt.annotate(txt, (number[i], fhe[i]))

plt.xlabel("Model Sizes")
plt.ylabel("File Size (Bytes)")
plt.legend(loc = 'best')
plt.savefig('model_comm.pdf', bbox_inches='tight')