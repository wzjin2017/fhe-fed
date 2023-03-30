import matplotlib.pyplot as plt
import numpy as np

labels = np.array(['Train', 'Comm:C-S', 'PlainAgg','Comm:S-C'])
colors = ['tab:red', 'tab:cyan', 'tab:green', 'tab:olive']
times = np.array([148.32,0.48895, 5.379,0.48895])
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
plt.savefig('pie-plain.pdf', bbox_inches='tight')