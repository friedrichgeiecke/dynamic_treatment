# -*- coding: utf-8 -*-
"""
Code for: DYNAMICALLY OPTIMAL TREATMENT ALLOCATION USING REINFORCEMENT LEARNING
by Karun Adusumilli, Friedrich Geiecke, and Claudio Schilter

This code: create figure F.2

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

#For subplot 1
folder="gridsearch\\run_2020_07_12_020610_0_0_0\\"
folder2="gridsearch\\run_2020_07_12_020610_0_0_1\\"
folder3="gridsearch\\run_2020_07_12_020610_0_0_2\\"

#For subplot 2
#folder="gridsearch\\run_2020_07_12_020612_0_1_0\\"
#folder2="gridsearch\\run_2020_07_12_020611_0_1_1\\"
#folder3="gridsearch\\run_2020_07_12_020612_0_1_2\\"

#For subplot 3
#folder="gridsearch\\run_2020_07_12_020612_0_2_0\\"
#folder2="gridsearch\\run_2020_07_12_035515_0_2_1\\"
#folder3="gridsearch\\run_2020_07_12_093815_0_2_2\\"



#For subplot 4
#folder="gridsearch\\run_2020_07_12_110412_1_0_0\\"
#folder2="gridsearch\\run_2020_07_12_115438_1_0_1\\"
#folder3="gridsearch\\run_2020_07_12_134129_1_0_2\\"

#For subplot 5
#folder="gridsearch\\run_2020_07_12_141528_1_1_0\\"
#folder2="gridsearch\\run_2020_07_12_153407_1_1_1\\"
#folder3="gridsearch\\run_2020_07_12_202605_1_1_2\\"

#For subplot 6
#folder="gridsearch\\run_2020_07_12_202605_1_2_0\\"
#folder2="gridsearch\\run_2020_07_12_210429_1_2_1\\"
#folder3="gridsearch\\run_2020_07_12_210429_1_2_2\\"



#For subplot 7
#folder="gridsearch\\run_2020_07_13_015504_2_0_0\\"
#folder2="gridsearch\\run_2020_07_13_021534_2_0_1\\"
#folder3="gridsearch\\run_2020_07_13_031704_2_0_2\\"

#For subplot 8
#folder="gridsearch\\run_2020_07_13_031734_2_1_0\\"
#folder2="gridsearch\\run_2020_07_13_040204_2_1_1\\"
#folder3="gridsearch\\run_2020_07_13_061033_2_1_2\\"

#For subplot 9
#folder="gridsearch\\run_2020_07_13_071833_2_2_0\\"
#folder2="gridsearch\\run_2020_07_13_104803_2_2_1\\"
#folder3="gridsearch\\run_2020_07_13_112703_2_2_2\\"


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

#For subplots 1-6
plt.ylim(0.8, 4.1) 

#For subplots 7-9
#plt.ylim(-0.7, 4.1) 

plt.xlim(-500, 10200) 


plt.plot(xaxis, yaxis/yaxis[0], linestyle='--', marker='o',linewidth=4, markersize=13, color=palette_used[8], label=r"$d_\nu=9$")
plt.plot(xaxis2, yaxis2/yaxis2[0], linestyle='--', marker='o',linewidth=4, markersize=13, color=palette_used[2], label=r"$d_\nu=11$")
plt.plot(xaxis3, yaxis3/yaxis3[0], linestyle='--', marker='o',linewidth=4, markersize=13, color=palette_used[1], label=r"$d_\nu=13$")

#For subplot 1
plt.title(r"$\alpha_\theta=0.5$, $\alpha_\nu=10^{-3}$")
#For subplot 2
#plt.title(r"$\alpha_\theta=5$, $\alpha_\nu=10^{-3}$")
#For subplot 3
#plt.title(r"$\alpha_\theta=50$, $\alpha_\nu=10^{-3}$")

#For subplot 4
#plt.title(r"$\alpha_\theta=0.5$, $\alpha_\nu=10^{-2}$")
#For subplot 5
#plt.title(r"$\alpha_\theta=5$, $\alpha_\nu=10^{-2}$")
#For subplot 6
#plt.title(r"$\alpha_\theta=50$, $\alpha_\nu=10^{-2}$")

#For subplot 7
#plt.title(r"$\alpha_\theta=0.5$, $\alpha_\nu=10^{-1}$")
#For subplot 8
#plt.title(r"$\alpha_\theta=5$, $\alpha_\nu=10^{-1}$")
#For subplot 9
#plt.title(r"$\alpha_\theta=50$, $\alpha_\nu=10^{-1}$")

#change the below to center right for subplot 8
plt.legend(loc="lower right")
plt.rc('font', family='serif')

plt.rc('font', family='serif')
plt.xlabel("Episodes")

#For subplots 7-9
#plt.yticks(np.arange(0,4.5,0.5))

#For subplots 1, 4, 7
plt.ylabel("Welfare")

#For subplots 2,3,5,6,8,9
#plt.tick_params(axis='y', colors=(0,0,0,0))

plt.rc('font', family='serif')

plt.show()

#Again, saving according to subplot
plt.savefig("gridsearch\\robustness_plot_appendix1.pdf", bbox_inches = 'tight', pad_inches = 0) 
#plt.savefig("gridsearch\\robustness_plot_appendix2.pdf", bbox_inches = 'tight', pad_inches = 0) 
#plt.savefig("gridsearch\\robustness_plot_appendix3.pdf", bbox_inches = 'tight', pad_inches = 0) 

#plt.savefig("gridsearch\\robustness_plot_appendix4.pdf", bbox_inches = 'tight', pad_inches = 0) 
#plt.savefig("gridsearch\\robustness_plot_appendix5.pdf", bbox_inches = 'tight', pad_inches = 0) 
#plt.savefig("gridsearch\\robustness_plot_appendix6.pdf", bbox_inches = 'tight', pad_inches = 0) 

#plt.savefig("gridsearch\\robustness_plot_appendix7.pdf", bbox_inches = 'tight', pad_inches = 0) 
#plt.savefig("gridsearch\\robustness_plot_appendix8.pdf", bbox_inches = 'tight', pad_inches = 0) 
#plt.savefig("gridsearch\\robustness_plot_appendix9.pdf", bbox_inches = 'tight', pad_inches = 0) 
