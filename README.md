# Dynamically Optimal Treatment Allocation Using Reinforcement Learning

Karun Adusumilli, Friedrich Geiecke, Claudio Schilter

### Step one

Obtain data as described in Stata code and then run Stata code to start setting up the data

Input: jtpa_kt.tab from Kitagawa & Tetenov's (2018) supplement to their paper (Econometric Society); expbif.dta from the original JTPA study (Upjohn Institute)

Code: 1_1_stata_code.do (see folder 1_dataset_part_1)

Output: the_table_ecma.csv; the_table_ecma_withClusters.csv; sincos_poisson_means_clusters9.csv


### Step two

Run python code to obtain rewards, finishing the datasets needed for the actor critic code

Input: the_table_ecma_withClusters.csv

Code: 2_1_rewards.py (see folder 2_dataset_part_2)

Output: data_lpe2.csv


### Step three

Run actor critic python code to obtain policy function etc (main step), either with all states, without budget and time, or without budget and time and age

Input: data_lpe2.csv; sincos_poisson_means_clusters9.csv (from steps one and two, but for convenience also provided in the subfolder "data")

Code:

- 3_1_dynamic_treatment (main setup)
- 3_2_dynamic_treatment_no_bt (no budget and time in policy)
- 3_3_dynamic_treatment_no_bt_no_age (no budget and time and age in policy)

Note: when running each of these three files, the scripts use 4 binary inputs. For example, execute:

python 3_1_dynamic_treatment.py 1 1 2 0

which runs the model with doubly robust rewards, a value function learning rate of 0.01, a policy function learning rate of 50, and a value basis function with 9 terms (see script and paper for details). In more detail, the four inputs determine

First input: 0 - standard ols rewards, 1 - doubly robust rewards

Second input: 0 - value function learning rate 0.001, 1 - value function learning rate 0.001, 2 - value function learning rate 0.1

Third input: 0 - policy function learning rate 0.5, 1 - policy function learning rate 5, 2 - policy function learning rate 50

Fourth input: 0 - value basis function parametrisation with 9 terms, 1 - value basis function parametrisation with 11 terms, 2 - value basis function parametrisation with 13 terms

(see folder 3_reinforcement_learning)

Main Output: policy_parameter_path.csv; transformed_rct_data_'reward'.csv; x_axis_for_rewards_'reward'.csv; y_axis_for_rewards_'reward'.csv
(in a separate folder for each specification; 'reward' being a placeholder for either d_rob_ols_Xfit or Rlr1)


### Step four

Run matlab code for EWM policy as benchmark

Input: jtpa_kt.mat (again from Kitagawa & Tetenov's (2018) supplement); age.csv (simply the sole age column of data_lpe2.csv (sorted by recid)); transformed_rct_data_'reward'.csv

Code: 4_1_kitagawa_tetenov_benchmark.m; 4_2_kitagawa_tetenov_benchmark_noage.m (see folder 4_EWM_benchmark)

Output: beta_dr_final_max025treated.csv; beta_ols_final_max025treated.csv; beta_kt_final_noage_389.csv


### Step five

Run python code to obtain figures and evaluate policies according to mean welfare

Input: output from steps one, three, and four

Code: 5_1_evaluation.py; all figure files 5_2_... (see folder 5_evaluation_and_figures)

Output: mean and standard deviation of welfare over N simulated episodes following the policies resulting from step three and four; all figures in the paper and appendix


### Step six

Optional histograms of individual level treatment effects/rewards which can be found in the applied companion paper
