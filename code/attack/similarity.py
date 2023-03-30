# pip install sewar
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
import cv2
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np

# with open('fhe_number.pickle', 'rb') as handle:
# 	t_cipher_list = pickle.load(handle)

# sns.set_style("whitegrid")
# Mean Squared Error (MSE)
# Peak Signal-to-Noise Ratio (PSNR)
# Universal Quality Image Index (UQI)
file_gt = "gt.png"
gt_data = cv2.imread(file_gt)
# print("MSE: ", mse(blur,org))
# print("UQI: ", uqi(blur, org))
# print("PSNR: ", psnr(blur, org))
msssim_list = []
uqi_list = []
vifp_list = []

for lid in range(10):
	if lid%2 == 0:
		filename = 'single_CIFAR100_900ite_['+ str(lid) + ']_' +'.png'
		img = cv2.imread(filename)
		msssim_list.append(msssim(img, gt_data).real)
		uqi_list.append(uqi(img, gt_data))
		vifp_list.append(vifp(img, gt_data))

number = [0, 2, 4, 6, 8]


plt.plot(number, msssim_list, color='tab:blue',linewidth=2,marker='s', markersize= 5,markevery=np.where(np.array(number) > 0, True, False),label='MSSSIM',linestyle='-')
plt.plot(number, uqi_list, color='tab:red',linewidth=2, marker='P', markersize= 5,markevery=np.where(np.array(number) > 0, True, False),label='UQI',linestyle='-')
plt.plot(number, vifp_list, color='tab:olive',linewidth=2, marker='o', markersize= 5,markevery=np.where(np.array(number) > 0, True, False),label='VIFP',linestyle='-')

plt.xlabel("Layer")
plt.ylabel("Recovered Image Score")
plt.legend(loc = 'best')
plt.savefig('attack_lenet.pdf', bbox_inches='tight')