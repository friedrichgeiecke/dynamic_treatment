# -*- coding: utf-8 -*-
"""
Code for: DYNAMICALLY OPTIMAL TREATMENT ALLOCATION USING REINFORCEMENT LEARNING
by Karun Adusumilli, Friedrich Geiecke, and Claudio Schilter

This code: generate figures 7.3, 7.4 and VI.3, VI.4
(depending on if DR or OLS folder is chosen initially)

prerequisites: having obtained the policy parameter paths

@author: Claudio Schilter
"""

import os
import pandas as pd
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.pyplot import colorbar, pcolor, show



##########
#Settings#
##########

# Main path to project
os.chdir("C:\\Users\\ClaudioSchilter\\Dropbox\\Reinforcement_learning\\")

# Folder where the RL policy is saved in
#DR (figure 7.3 and 7.4)
folder="fabian_outcomes_to_analyse\\paper\\run_2020_07_21_001221_1_1_0\\"
#OLS (figure VI.3 and VI.4)
#folder="fabian_outcomes_to_analyse\\paper\\run_2020_07_21_00159_1_1_0\\"

# Style of graphs
plt.style.use('seaborn')
plt.rc('font', family='serif')
palette_used = sns.color_palette("husl", 12)
sns.set(font_scale=2)
mesh_fineness = 0.001
total_budget = 1
sns.set(font_scale=3)
plt.rc('font', family='serif')

# Load the dynamic policy function (see other Python code)
dataset = pd.read_csv(folder+"policy_parameter_path.csv")
episode_counter=list(dataset['episode'])
del dataset['episode']
#flipping all parameter signs as in the paper we have exp(h) and in the code exp(-h)
dataset=-dataset

#############################################
#Convergence of Policy Function Coefficients#
#############################################

## Number of rows in the dataset (needed later to index)
epi=dataset.shape[0]

# Naming the coefficients
name_list=['Constant', 'Age', 'Education', 'Previous Earnings','Age x Budget', 'Education x Budget', 'Previous Earnings x Budget','Age x cos_2pi_Time', 'Education x cos_2pi_Time', 'Previous Earnings x cos_2pi_Time', 'Budget', 'cos_2pi_Time']
name_list2=['Constant', 'Age', 'Education', 'Previous Earnings','Age x Budget', 'Education x Budget', 'Previous Earnings x Budget','Age x $cos(2 \pi t)$', 'Education x $cos(2 \pi t)$', 'Previous Earnings x $cos(2 \pi t)$', 'Budget', '$cos(2 \pi t)$']

# Convergence of each coefficient separately (not done in paper)
for x in np.arange(0, 12, 1):
    plt.figure(x)
    plt.plot(episode_counter,dataset.iloc[:,x])
    plt.title(name_list[x])
    plt.rc('font', family='serif')
    plt.show
    plt.savefig(folder+"policy_parm_plots_"+name_list[x]+".pdf")



# Convergence of all coefficients
for x in np.arange(0, 12, 1):
    plt.figure(42, figsize=(25,14))
    matplotlib.rcParams.update({'font.size': 25})
    plt.plot(episode_counter,dataset.iloc[:,x], label=name_list2[x], color=palette_used[x], linewidth=3)
    plt.xlabel("Episodes")
    plt.ylabel("Parameter Value")
    plt.rc('font', family='serif')
#    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.legend(bbox_to_anchor=(0.5,-0.1), loc="upper center",ncol=3, borderaxespad=0)
    plt.rc('font', family='serif')
    plt.show

plt.savefig(folder+"policy_parm_plots.pdf", bbox_inches='tight')


# Convergence of the ratio of all other coefficients relative to the constant
# ("zoomed in" by limiting y-axis between -5 and 5)
for x in np.arange(1, 12, 1):
    plt.figure(742, figsize=(25,14))
    matplotlib.rcParams.update({'font.size': 25})
    plt.plot(episode_counter,dataset.iloc[:,x]/dataset.iloc[:,0], label=name_list2[x], color=palette_used[x], linewidth=3)
    plt.ylim(-5, 5)
    plt.xlabel("Episodes")
    plt.ylabel("Parameter Value Relative to Intercept")
    plt.rc('font', family='serif')
#    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    plt.legend(bbox_to_anchor=(0.5,-0.1), loc="upper center",ncol=3, borderaxespad=0)
    plt.rc('font', family='serif')
    plt.show

#plt.savefig(folder+"policy_parm_ratios_plots.pdf")
plt.savefig(folder+"policy_parm_ratios_plots_zoom_in.pdf", bbox_inches='tight')



###################################################################
#Behavior of Final Policy Function with Budget and Time (Heatmaps)#
###################################################################

# Defining grid
t=np.arange(0, 1, mesh_fineness)
b=np.arange(0, total_budget, mesh_fineness)
b, t = np.meshgrid(b, t)

# Select style
plt.rcParams.update({'font.size': 22})
husl_map_part_used = matplotlib.colors.ListedColormap(sns.color_palette(sns.color_palette("husl", 100)[:70][::-1]).as_hex())

 

#age
Z = dataset.iloc[(epi-1),1]+dataset.iloc[(epi-1),4]*b+dataset.iloc[(epi-1),7]*np.cos(2*np.pi*t)
plt.figure(77, figsize=(14,18))
plt.title('Age Coefficient')
plt.xlabel('Time')
plt.ylabel('Budget')
plt.rc('font', family='serif')
pcolor(t, b, Z, cmap = husl_map_part_used)
plt.colorbar(orientation="horizontal")
plt.show

plt.savefig(folder+"age_coef.png", bbox_inches = 'tight',
    pad_inches = 0)

#education
Z = dataset.iloc[(epi-1),2]+dataset.iloc[(epi-1),5]*b+dataset.iloc[(epi-1),8]*np.cos(2*np.pi*t)
plt.figure(777, figsize=(14,18))
plt.title('Education Coefficient')
plt.xlabel('Time')
plt.yticks([], [])
plt.rc('font', family='serif')
pcolor(t, b, Z, cmap = husl_map_part_used)
plt.colorbar(orientation="horizontal")
plt.show

plt.savefig(folder+"edu_coef.png",bbox_inches = 'tight',
    pad_inches = 0)

#prev earnings
Z = dataset.iloc[(epi-1),3]+dataset.iloc[(epi-1),6]*b+dataset.iloc[(epi-1),9]*np.cos(2*np.pi*t)
plt.figure(7777, figsize=(14,18))
plt.title('Previous Earnings Coefficient')
plt.xlabel('Time')
plt.yticks([], [])
plt.rc('font', family='serif')
pcolor(t, b, Z, cmap = husl_map_part_used)
plt.colorbar(orientation="horizontal")
plt.show

plt.savefig(folder+"prev_earn_coef.png",bbox_inches = 'tight',
    pad_inches = 0)




