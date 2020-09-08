# -*- coding: utf-8 -*-
"""
Code for: DYNAMICALLY OPTIMAL TREATMENT ALLOCATION USING REINFORCEMENT LEARNING
by Karun Adusumilli, Friedrich Geiecke, and Claudio Schilter

This code: produce figure F.1 (illustrating arrival of individuals from different clusters)

Prerequisites: stata code ran
"""
#visualizing clusters frequencies over time

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
import pandas as pd
import os


data_directory = '/Users/ClaudioSchilter/Dropbox/Reinforcement_learning/evaluation/data/'
poisson_table = pd.read_csv(os.path.join(data_directory, "sincos_poisson_means_clusters9.csv"), sep='\t')

Workdays=np.arange(1, 252, 1)

Cluster1_sincos=poisson_table['estimate'][2]+poisson_table['estimate'][0]*np.sin(2*np.pi*Workdays/252)+poisson_table['estimate'][1]*np.cos(2*np.pi*Workdays/252)
Cluster2_sincos=poisson_table['estimate'][5]+poisson_table['estimate'][3]*np.sin(2*np.pi*Workdays/252)+poisson_table['estimate'][4]*np.cos(2*np.pi*Workdays/252)
Cluster3_sincos=poisson_table['estimate'][8]+poisson_table['estimate'][6]*np.sin(2*np.pi*Workdays/252)+poisson_table['estimate'][7]*np.cos(2*np.pi*Workdays/252)
Cluster4_sincos=poisson_table['estimate'][11]+poisson_table['estimate'][9]*np.sin(2*np.pi*Workdays/252)+poisson_table['estimate'][10]*np.cos(2*np.pi*Workdays/252)





plt.style.use('seaborn')
plt.rc('font', family='serif')
palette_used = sns.color_palette("husl", 12)

sns.set(font_scale=2)


plt.figure(figsize=(20,14))
matplotlib.rcParams.update({'font.size': 22})
plt.rc('font', family='serif')

plt.plot(Workdays, Cluster1_sincos, color=palette_used[8], linewidth=5, dashes=[6, 2], label='Cluster 1')
plt.plot(Workdays, Cluster2_sincos, color=palette_used[4], linewidth=5, dashes=[2, 2], label='Cluster 2')
plt.plot(Workdays, Cluster3_sincos, color=palette_used[9], linewidth=5, label='Cluster 3')
plt.plot(Workdays, Cluster4_sincos, color=palette_used[1], linewidth=5, dashes=[4, 6], label='Cluster 4') 
plt.xlabel('Workdays')
plt.ylabel('Log(Poisson Mean)')
plt.rc('font', family='serif')

plt.legend(prop={'size': 20})
plt.rc('font', family='serif')

plt.savefig(data_directory+"ClusterGraph.pdf", bbox_inches = 'tight', pad_inches = 0)
plt.show()
