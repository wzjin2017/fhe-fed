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


fhe = [0.216,
2.792,
0.586,
0.619,
1.195,
2.456,
9.481,
19.950,
37.555,
46.672,
86.098,
112.504,
136.914]
plain = [0.001,
0.700,
0.010,
0.011,
0.033,
0.058,
1.031,
1.100,
2.925,
5.379,
19.921,
17.739,
19.674]

nvidia = [0.212365071,
0.90594848,
0.3772704601,
0.4061382612,
2.00879852,
3.795688073,
8.305898507,
25.45245687,
48.78017847,
55.55209533,
120.8812542,
141.1431133,
141.1431133
]

p10 = [0.02286711895,
0.4885876681,
0.06730369127,
0.07156748673,
0.1311670355,
0.2979262544,
1.875507048,
2.984632613,
6.387871424,
9.508372378,
26.53908812,
27.215495,
31.39823451]

p50 = [0.1086135489,
1.512325776,
0.2977575082,
0.3150764855,
0.6037961072,
1.257028103,
5.255515987,
10.5247599,
20.23976072,
26.02557218,
53.009703,
65.12157003,
78.29432845]


plt.plot(number, fhe, color='tab:blue',linewidth=2,marker='s', markersize= 5,markevery=np.where(np.array(number) > 0, True, False),label='Ours (Full Encryption)',linestyle='-')
plt.plot(number, nvidia, color='tab:red',linewidth=2, marker='P', markersize= 5,markevery=np.where(np.array(number) > 0, True, False),label='FLARE',linestyle='-')
plt.plot(number, plain, color='tab:olive',linewidth=2, marker='o', markersize= 5,markevery=np.where(np.array(number) > 0, True, False),label='Plaintext',linestyle='-')
plt.plot(number, p10, color='tab:green',linewidth=2, marker='v', markersize= 5,markevery=np.where(np.array(number) > 0, True, False),label='Ours (10% Encryption)',linestyle='-')
plt.plot(number, p50, color='tab:purple',linewidth=2, marker='1', markersize= 5,markevery=np.where(np.array(number) > 0, True, False),label='Ours (50% Encryption)',linestyle='-')
# for i, txt in enumerate(models):
#     plt.annotate(txt, (number[i], fhe[i]))

plt.xlabel("Model Sizes")
plt.ylabel("Execution Time (s)")
plt.legend(loc = 'best')
plt.savefig('model_comp_opt.pdf', bbox_inches='tight')