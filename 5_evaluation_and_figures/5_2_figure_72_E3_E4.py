# -*- coding: utf-8 -*-
"""
Code for: DYNAMICALLY OPTIMAL TREATMENT ALLOCATION USING REINFORCEMENT LEARNING
by Karun Adusumilli, Friedrich Geiecke, and Claudio Schilter

This code: create Figures 7.2, E.3, and E.4

Prerequisite: runs completed for DR rewards (with and without budget/time) and std OLS rewards
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

#For Figure 7.2 (DR rewards)
folder="fabian_outcomes_to_analyse\\paper\\run_2020_07_21_001221_1_1_0\\"
folder2="fabian_outcomes_to_analyse\\paper\\run_2020_07_22_235745_1_1_0_without_budget_time\\"

#For Figure E.4 (std OLS rewards)
#folder="fabian_outcomes_to_analyse\\paper\\run_2020_07_21_001159_1_1_0\\"

#For Figure E.3 (no age)
#folder="fabian_outcomes_to_analyse\\paper\\run_2020_07_23_000146_1_1_0_without_budget_time_age\\"


#For Figures 7.2 and E.3
xaxis = pd.read_csv(folder+"x_axis_for_rewards_d_rob_ols_Xfit.csv",header=None)
yaxis = pd.read_csv(folder+"y_axis_for_rewards_d_rob_ols_Xfit.csv",header=None)

#For Figure E.4
#xaxis = pd.read_csv(folder+"x_axis_for_rewards_Rlr1.csv",header=None)
#yaxis = pd.read_csv(folder+"y_axis_for_rewards_Rlr1.csv",header=None)


xaxis = np.array(xaxis.iloc[:,0])
yaxis = np.array(yaxis.iloc[:,0])


#For Figure 7.2 only
xaxis2 = pd.read_csv(folder2+"x_axis_for_rewards_d_rob_ols_Xfit.csv",header=None)
xaxis2 = np.array(xaxis2.iloc[:,0])
yaxis2 = pd.read_csv(folder2+"y_axis_for_rewards_d_rob_ols_Xfit.csv",header=None)
yaxis2 = np.array(yaxis2.iloc[:,0])


notnans = ~np.isnan(yaxis)

#For Figure 7.2 only
notnans2 = ~np.isnan(yaxis2)


yaxis = yaxis[notnans]
xaxis = xaxis[notnans]

#For Figure 7.2 only
yaxis2 = yaxis2[notnans2]
xaxis2 = xaxis2[notnans2]

#For Figure 7.2 only: make paths the same length by just showing the basis one and then 160 points
yaxis = yaxis[0:161]
xaxis = xaxis[0:161]
yaxis2 = yaxis2[0:161]
xaxis2 = xaxis2[0:161]

#KT for doubly robust (Figure 7.2):
KT_welfare = 0.0098917

#KT for ols (Figure E.4):
#KT_welfare = 0.4285785

#KT no age  (Figure E.3):
#KT_welfare = 0.0077965

plt.style.use('seaborn')
plt.rc('font', family='serif')

palette_used = sns.color_palette("husl", 12)
sns.set(font_scale=3)

plt.figure(123, figsize=(35,10))
plt.rc('font', family='serif')

#labelling and saving set up for Figure 7.2 (change accordingly for E.3 and E.4)
plt.plot(xaxis, yaxis/yaxis[0], linestyle='--', marker='o',linewidth=4, markersize=13, color=palette_used[8], label="Policy Function")
plt.plot(xaxis2, yaxis2/yaxis2[0], linestyle='--', marker='o',linewidth=4, markersize=13, color=palette_used[2], label="Restricted Policy Function")
plt.axhline(KT_welfare/yaxis[0], linestyle='-', linewidth=4, color=palette_used[4], label="EWM Policy")
plt.legend(loc="lower right")
plt.rc('font', family='serif')
plt.xlabel("Episodes")
plt.ylabel("Welfare")
plt.rc('font', family='serif')

plt.show()

plt.savefig(folder+"welfare_plot_double_new.pdf", bbox_inches = 'tight', pad_inches = 0) 
