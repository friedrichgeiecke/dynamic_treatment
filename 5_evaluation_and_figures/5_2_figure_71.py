# -*- coding: utf-8 -*-
"""
Code for: DYNAMICALLY OPTIMAL TREATMENT ALLOCATION USING REINFORCEMENT LEARNING
by Karun Adusumilli, Friedrich Geiecke, and Claudio Schilter

This code: create Figure 7.1

Run separately for each subplot (comment/uncomment code lines accordingly)

Prerequisite: gridsearch runs completed
"""

# Importing modules
import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')
plt.style.use('seaborn')
import numpy as np
import pandas as pd
import os


# Main path to project
os.chdir("C:\\Users\\ClaudioSchilter\\Dropbox\\Reinforcement_learning\\")

# Folder where the RL policy is saved in
folder="gridsearch\\run_2020_07_12_141528_1_1_0\\"

#use for subplot 1
folder2="gridsearch\\run_2020_07_12_020612_0_1_0\\"
folder3="gridsearch\\run_2020_07_13_031734_2_1_0\\"

#use for subplot 2
#folder2="gridsearch\\run_2020_07_12_110412_1_0_0\\"
#folder3="gridsearch\\run_2020_07_12_202605_1_2_0\\"

#use for subplot 3
#folder2="gridsearch\\run_2020_07_12_153407_1_1_1\\"
#folder3="gridsearch\\run_2020_07_12_202605_1_1_2\\"

#import for normalizing term (from the random policy)
folder_paper="fabian_outcomes_to_analyse\\paper\\run_2020_07_21_001221_1_1_0\\"
yrand= pd.read_csv(folder_paper+"y_axis_for_rewards_d_rob_ols_Xfit.csv",header=None)[0][0]


xaxis = pd.read_csv(folder+"x_axis_for_rewards_d_rob_ols_Xfit.csv",header=None)
xaxis = np.array(xaxis.iloc[:,0])

xaxis2 = pd.read_csv(folder2+"x_axis_for_rewards_d_rob_ols_Xfit.csv",header=None)
xaxis2 = np.array(xaxis2.iloc[:,0])

xaxis3 = pd.read_csv(folder3+"x_axis_for_rewards_d_rob_ols_Xfit.csv",header=None)
xaxis3 = np.array(xaxis3.iloc[:,0])

yaxis = pd.read_csv(folder+"y_axis_for_rewards_d_rob_ols_Xfit.csv",header=None)
yaxis = np.array(yaxis.iloc[:,0])

yaxis2 = pd.read_csv(folder2+"y_axis_for_rewards_d_rob_ols_Xfit.csv",header=None)
yaxis2 = np.array(yaxis2.iloc[:,0])

yaxis3 = pd.read_csv(folder3+"y_axis_for_rewards_d_rob_ols_Xfit.csv",header=None)
yaxis3 = np.array(yaxis3.iloc[:,0])


notnans = ~np.isnan(yaxis)
notnans2 = ~np.isnan(yaxis2)
notnans3 = ~np.isnan(yaxis3)


yaxis = yaxis[notnans]
xaxis = xaxis[notnans]
yaxis2 = yaxis2[notnans2]
xaxis2 = xaxis2[notnans2]
yaxis3 = yaxis3[notnans3]
xaxis3 = xaxis3[notnans3]

yaxis[0] = yrand
yaxis2[0] = yrand
yaxis3[0] = yrand


plt.style.use('seaborn')
plt.rc('font', family='serif')

palette_used = sns.color_palette("husl", 12)
sns.set(font_scale=3) #or 2

plt.figure(123, figsize=(15,10))
plt.rc('font', family='serif')

plt.ylim(0.8, 4.1) 

#use for subplot 1
plt.plot(xaxis, yaxis/yaxis[0], linestyle='--', marker='o',linewidth=4, markersize=13, color=palette_used[8], label=r"$\alpha_\nu=10^{-2}$")
plt.plot(xaxis2, yaxis2/yaxis2[0], linestyle='--', marker='o',linewidth=4, markersize=13, color=palette_used[2], label=r"$\alpha_\nu=10^{-3}$")
plt.plot(xaxis3, yaxis3/yaxis3[0], linestyle='--', marker='o',linewidth=4, markersize=13, color=palette_used[1], label=r"$\alpha_\nu=10^{-1}$")

#use for subplot 2
#plt.plot(xaxis, yaxis/yaxis[0], linestyle='--', marker='o',linewidth=4, markersize=13, color=palette_used[8], label=r"$\alpha_\theta=5$")
#plt.plot(xaxis2, yaxis2/yaxis2[0], linestyle='--', marker='o',linewidth=4, markersize=13, color=palette_used[2], label=r"$\alpha_\theta=0.5$")
#plt.plot(xaxis3, yaxis3/yaxis3[0], linestyle='--', marker='o',linewidth=4, markersize=13, color=palette_used[1], label=r"$\alpha_\theta=50$")

#use for subplot 3
#plt.plot(xaxis, yaxis/yaxis[0], linestyle='--', marker='o',linewidth=4, markersize=13, color=palette_used[8], label=r"$d_\nu=9$")
#plt.plot(xaxis2, yaxis2/yaxis2[0], linestyle='--', marker='o',linewidth=4, markersize=13, color=palette_used[2], label=r"$d_\nu=11$")
#plt.plot(xaxis3, yaxis3/yaxis3[0], linestyle='--', marker='o',linewidth=4, markersize=13, color=palette_used[1], label=r"$d_\nu=13$")




plt.legend(loc="lower right")
plt.rc('font', family='serif')
plt.rc('font', family='serif')
plt.xlabel("Episodes")

#only used for subplot 1
plt.ylabel("Welfare")

#used for subplots 2 and 3
#plt.tick_params(axis='y', colors=(0,0,0,0))

plt.rc('font', family='serif')

plt.show()

#use for subplot 1
plt.savefig("gridsearch\\robustness_plot_valuerate.pdf", bbox_inches = 'tight', pad_inches = 0) 
#use for subplot 2
#plt.savefig("gridsearch\\robustness_plot_policyrate.pdf", bbox_inches = 'tight', pad_inches = 0) 
#use for subplot 3
#plt.savefig("gridsearch\\robustness_plot_valuefunction.pdf", bbox_inches = 'tight', pad_inches = 0) 
