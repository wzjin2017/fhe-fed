import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")

x = ['MAR (HE)', 'SAR (HE)', 'IB (HE)', 'MAR(Non)', 'SAR (Non)', 'IB (Non)']
comm = np.array([103.713,
2.733,
0.316,6.269,
0.165,
0.019])
total = np.array([298.703,
197.723,
195.306, 159.968,
153.864,
153.718])
rest = total-comm

percs = ['34.72%',
'1.38%',
'0.16%', '3.92%',
'0.11%',
'0.01%']

for index, value in enumerate(percs):
    plt.text(index-0.3, total[index]+3,
             value)

plt.bar(x, rest, color='tab:green')
plt.bar(x, comm, bottom=rest, color='tab:red')

plt.legend(["Others", "Communication"])
plt.xlabel("Bandwidths")
plt.ylabel("Time Elapsed (s)")
plt.savefig('comm_bar.pdf', bbox_inches='tight')