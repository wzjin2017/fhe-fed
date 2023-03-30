import matplotlib.pyplot as plt
import numpy as np

labels = np.array(['Init (w/ Mask \n Agreement)', 'Train', 'Enc', 'Comm:C-S', 'FHEAgg','Comm:S-C', 'Dec'])
colors = ['tab:brown', 'tab:red', 'tab:cyan', 'tab:purple', 'tab:green', 'tab:olive', 'tab:orange']
times = np.array([0.009775280952+0.08004260063171387, 148.32, 9.982236147 * 0.12, 8.116 * 0.12 ,17.47818494 *0.12 ,8.116*0.12,19.20012999*0.12])
porcent = 100.*times/times.sum()
patches, texts = plt.pie(times,colors=colors, startangle=90)

labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(labels, porcent)]

sort_legend = True
if sort_legend:
    patches, labels, dummy =  zip(*sorted(zip(patches, labels, times),
                                          key=lambda labels: labels[2],
                                          reverse=True))
plt.legend(patches, labels, loc='center left', bbox_to_anchor=(-0.35, .5), fontsize=12)
plt.axis('equal')
plt.savefig('pie-he-opt.pdf', bbox_inches='tight')