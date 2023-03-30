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
0.033,
0.058,
1.031,
1.100,
2.925,
5.379,
19.921,
17.739,
19.674]

comp = [150.85,
12.00,
60.46,
91.82,
42.23,
9.20,
18.14,
12.84,
8.68,
4.32,
6.34,
6.96]
comm = [240.83,
10.10,
17.05,
16.70,
16.66,
16.45,
16.61,
16.60,
16.58,
16.61,
16.62,
16.62]

fhe_file = [272384,
544768,
5447680,
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
3293785,
6656175,
13408318,
50309716,
87328827,
102543033,
223166212,
345633211,
438009167]


plt.plot(number, comm, color='tab:blue',linewidth=2,marker='s', markersize= 5,markevery=np.where(np.array(number) > 0, True, False),label='Communication',linestyle='-')
plt.plot(number, comp, color='tab:red',linewidth=2, marker='P', markersize= 5,markevery=np.where(np.array(number) > 0, True, False),label='Computation',linestyle='-')
# for i, txt in enumerate(models):
#     plt.annotate(txt, (number[i], fhe[i]))

plt.xlabel("Model Sizes")
plt.ylabel("Overhead Fold Ratio(s)")
plt.legend(loc = 'best')
plt.savefig('model_ratio.pdf', bbox_inches='tight')