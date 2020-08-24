"""
Code for: DYNAMICALLY OPTIMAL TREATMENT ALLOCATION USING REINFORCEMENT LEARNING
by Karun Adusumilli, Friedrich Geiecke, and Claudio Schilter

This code: a faster evaluation to obtain only the average welfare (but not all other information in Figure 7.5)
Both our policy function and EMW can be evaluated (choose accordingly in part 2)
Also the static policy can be evaluated (choose accordingly in the basis function of theta and part 2)
Finally, the case with no age can be evaluated - several changes are needed and indicated throughout the code

Part 1: Functions

Part 2: Simulation


Prerequisite: stata code, rewards.py and actor critic python code ran
"""
# Importing modules
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.style.use('seaborn')
import numpy as np
import pandas as pd
import os
import math
from math import pi
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
import time as time_package
import sys
import logging
import datetime




"""
Part 1: Functions
"""

# Creating a class for the function approximators
class function_approximators():

    # Initialising values
    def __init__(self, number_of_states_theta, number_of_states_w):

        # Initialise parameters
        self.number_of_states_theta = number_of_states_theta
        self.number_of_states_w = number_of_states_w
        self.delta_prime = 0
        self.theta = [0] * (self.number_of_states_w) # will be replaced with a shared object between subprocesses
        self.w = [0] * self.number_of_states_w # ditto


    def return_basis_function_theta(self, s0, budget_left, time, T):

        s_theta = np.vstack((1, s0, s0*budget_left, s0*np.cos(2*pi*time), budget_left, np.cos(2*pi*time)))
        
        #If evaluate instead the static policy (with no terms that include budget or time):
#        s_theta = np.vstack((1, s0))
        #If do that, note that also need to change "number of states theta" in part 2


        return s_theta


    def return_basis_function_w(self, budget_left, time, T):

        s_w =  np.vstack((  (budget_left)*(T-time), (budget_left)*(T-time)**2, (budget_left)**2*(T-time), ((budget_left)*(T-time))**2,
                            (budget_left)*np.sin(pi*time), (budget_left)*np.sin(2*pi*time), (budget_left)**2*np.sin(pi*time),                                         ((budget_left)**2*np.sin(2*pi*time)), (budget_left)**3*(T-time)   ))

        return s_w


    # Policy function
    def pi(self, s_theta):

        h = np.dot(s_theta[:,0], self.theta[:]) # np dot products can deal with lists. Have to use [:] here because theta will be a shared list object from the multiprocessing module

        treat_prob = 1 - (1/(1 + np.exp(-h)))

        return treat_prob


    # Value function
    def v(self, s_w):

        output = np.dot(s_w[:,0], self.w[:])

        return output


    # Gradient of the log policy
    def gradient_log_pi(self, s_theta, a):

        """

        Gives the gradient vector for the log policy pi given a certain action a
        (would be a Jacobian if all actions were to be considered)

        """

        output = (-(a-self.pi(s_theta))*s_theta)[:,0]

        return output


    # TD error
    def update_delta_prime(self, R, gamma, s_w, s_prime_w, terminal_flag):

        """

        In the final period, the TD error is set to zero

        """

        if terminal_flag == False:

            self.delta_prime = R + gamma*self.v(s_prime_w) - self.v(s_w)

        else:

            self.delta_prime = 0

    # Compute the update for the value function
    def compute_value_parameter_update(self, alpha_w, s_w):

        """

        Note:

            - s_theta is the state of the policy function
            - s_w is the state of the value function

            - self.delta_prime is just a float, which is updated by the delta prime update function

        """

        # Updating parameter values of the estimator
        return(alpha_w*self.delta_prime*s_w[:,0])


    # Compute update for the policy function
    def compute_policy_parameter_update(self, alpha_theta, s_theta, a, I):

        # Updating parameter values of the estimator


        return(alpha_theta*I*self.delta_prime*self.gradient_log_pi(s_theta, a))




# This function simulates the entire environment in a sub process
def eval(EE,
         policy_parameters,
         description,
         sampled_actions,
         seed,
         output_directory):


    """

    EE: Number of evaluation episodes (int)

    policy_parameters: A list of policy parameter values (list)

    description: One string without spaces, e.g. "KT_times_100" (str)

    sampled_actions: Indicator whether the actions were sampled or a 0.5 treshold was used (Bool)

    seed: Pseudo random number seed (int)

    output_directory: Directory name where output shall be saved (str)

    """

    log_status.info(f"\n\nStarting evaluation ({EE} episodes) ...\n")

    # Fixing the seed
    np.random.seed(seed=seed)

    # Setting the parameters in the agent instance
    function_approximators_object.theta[:] = policy_parameters

    # Evaluation rewards achieved
    cumulative_episode_rewards = []

    # Begin the loop over episodes
    ee = 0
    while ee < EE:

        # Restarting period time
        tic = time_package.time()

        # Next, intialise cumulative reward, time, and budget
        cumulative_episode_reward = 0
        time = 0
        budget_left = deepcopy(total_budget)

        # Initialising I (pre-multiplied by gamme for discounting)
        I = 1

        # People considered and treated in episode
        people_considered_for_treatment = 0
        people_treated = 0

        # Initialise with first person from first cluster
        s0 = cluster_dict[0].loc[0,['age', 'education', 'prev_earnings']].values.reshape(3,1)
        
        #in case of no age, change above to:
#        s0 = cluster_dict[0].loc[0,['education', 'prev_earnings']].values.reshape(2,1)


        # Building basis function
        s_theta = function_approximators_object.return_basis_function_theta(s0 = s0, budget_left = budget_left, time = time, T = T)

        # Treatment reward
        s_R_if_treated = cluster_dict[0].loc[0,reward_column_name]

        # Re-setting the terminal flag to False
        terminal_flag = False

        # Loop over people within episode as long as we are not in a terminal episode
        while terminal_flag == False:

            # Sample action from current policy
            treatment_probability = function_approximators_object.pi(s_theta)

            if sampled_actions == True:

                a = np.random.binomial(1, treatment_probability)

            elif sampled_actions == False:

                a = 1 * (treatment_probability > 0.5)

            # Set reward to zero if not treated and otherwise to 1
            if a == 0:

                R = 0

            # Otherwise update reward, people treated, and budget left
            elif a == 1:

                R = s_R_if_treated

                people_treated += 1

                budget_left -= cost_per_treatment

            # Update episode reward
            cumulative_episode_reward += I*R

            # Updating the people count within the episode
            people_considered_for_treatment += 1

            # For each cluster, draw the expected number of people arriving per 100th of a day
            lambda_vector = (np.exp(alpha_vector + (beta_vector * np.sin(time)) + (gamma_vector * np.cos(time))) / lambda_normalisation).reshape(4,1)

            # Summed lambda vector: The expected number of people (irrespective of clusters) arriving in a 100th of a day
            summed_lambda_vector = np.sum(lambda_vector)

            # Updating time (t' and budget_left' are already needed for the function approximator updates of this period)

            # The lambda fed into an exponential from which now a time increment will be drawn. As lambda is of the scale arrivals per 100th day,
            # the drawn time will be in the same time units. So a drawn value could be 50 as in 50 time units or half a day 50/100. Yet, our time
            # is in years, so we need to transform the drawn time into years. Hence we divide it first by 100 to have days, and then by 252 to have years.
            # This is why time_grid_normalisation has a value of 25200
            delta_time = (math.ceil(np.random.exponential(scale=1/summed_lambda_vector, size=None)) / time_grid_normalisation)
            time = time + delta_time

            # Creating discounting factor gamma that is time increment specific (discounting is actually only updated after the function approximator updates of that episode)
            gamma = np.exp(- beta * delta_time)

            # Drawing the cluster from which the next person arrives
            cluster = np.dot(np.array([0, 1, 2, 3]), np.random.multinomial(1, pvals = (lambda_vector/summed_lambda_vector).reshape(4,)))

            # Drawing the next individual
            sample_person_index = int(np.random.choice(range(0,cluster_dict[cluster].shape[0]), size=1))
            s0_prime = cluster_dict[cluster].loc[sample_person_index,['age', 'education', 'prev_earnings']].values.reshape(3,1)
            #in case of no age, change above to:    
#            s0_prime = cluster_dict[cluster].loc[sample_person_index,['education', 'prev_earnings']].values.reshape(2,1)


            # Building basis functions
            s_theta_prime = function_approximators_object.return_basis_function_theta(s0 = s0_prime, budget_left = budget_left, time = time, T = T)

            # Drawing the associated reward
            s_prime_R_if_treated = cluster_dict[cluster].loc[sample_person_index,reward_column_name]

            # Updating the cumulative discount factor I
            I = gamma*I

            # Reassigning state vectors
            s_theta = s_theta_prime
            s_R_if_treated = s_prime_R_if_treated


            # Updating the terminal flag
            if budget_left < cost_per_treatment or time >= T:

                cumulative_episode_rewards.append(cumulative_episode_reward)

                terminal_flag = True
                toc = time_package.time()

                if (ee+1) % 1 == 0:

                    log_status.info(f"Ended evaluation episode {ee+1}, current mean episode reward during evaluation {np.mean(cumulative_episode_rewards):.4f}. This episode took approximately {(toc-tic):.2f} seconds. {people_considered_for_treatment} people were considered for treatment, {people_treated} were treated, until the budget was {budget_left:.4f}, and time was {time:.4f}.")

                # Incrementing the episode by 1
                ee += 1

        # Resetting a couple of key assignments after the episode, to be sure we become aware when they are accidentaly reused
        # Make code more functional in the future to avoid such hacks
        s0 = None
        s0_prime = None
        s_theta = None
        s_theta_prime = None
        s_prime_R_if_treated = None
        s_R_if_treated = None

    # Save evaluation rewards once done
    np.savetxt(os.path.join(output_directory, f"cumulative_episode_returns_{description}_sampled_{sampled_actions}.csv"),
               cumulative_episode_rewards,
               delimiter=',')

    # Update that the script is finished
    log_status.info(f'\nEvaluation completed for {EE} episodes. Sum of all cumulative rewards: {np.sum(cumulative_episode_rewards):.4f}, mean cumulative reward: {np.mean(cumulative_episode_rewards):.4f}, standard deviation: {np.std(cumulative_episode_rewards):.4f}.')




"""
Part 1: Simulation
"""


if __name__ == "__main__":



    ##
    ## 1. Parameters
    ##

    # Path. This folder has to contain another folder "data" with the datasets "data_lpe2.csv" (output from rewards.py) and "sincos_poisson_means_clusters9.csv" (output from stata code)
    local_directory = '/Users/ClaudioSchilter/Dropbox/Reinforcement_learning/evaluation/'


    # Evaluation episodes
    EE = 1000

    # Policy
    # Choose if actions are sampled (as in the standard case of our policy) or not (as in the deterministic version of our policy)
    # Always choose False for EWM by Kitagawa & Tetenov
    sampled_actions = False
    policy_description = "ols_110"
    
    #In case of evaluating our policy, use policy parameters of either the policy based on doubly robust or OLS rewards
    dataset = pd.read_csv("C:\\Users\\ClaudioSchilter\\Dropbox\\Reinforcement_learning\\fabian_outcomes_to_analyse\\paper\\run_2020_07_21_001159_1_1_0\\policy_parameter_path.csv")
    del dataset['episode']
    policy_parameters = dataset[-1:].values.tolist()[0]

    # Alternatively, in case of EWM, choose beta from EWM that is either based on doubly robust or OLS rewards

#    dataset = pd.read_csv("C:\\Users\\ClaudioSchilter\\Dropbox\\Reinforcement_learning\\Kitagawa_Tetenow_Replication_files\\beta_ols_final_max025treated.csv",header=None)
#    data_directory = os.path.join(local_directory, 'data')
#    df = pd.read_csv(os.path.join(data_directory, "data_lpe2.csv"))
#    #convert policy since KT use different standardization
#    mean_age = np.mean(df["age"])
#    mean_edu = np.mean(df["education"])
#    mean_pe = np.mean(df["prev_earnings"])
#    sd_age = np.std(df["age"])
#    sd_edu = np.std(df["education"])
#    sd_pe = np.std(df["prev_earnings"])
#    max_age = np.max(df["age"])
#    max_edu = np.max(df["education"])
#    max_pe = np.max(df["prev_earnings"])
#    
#    dataset[0][1] = (-1)*dataset[0][1]/max_age*sd_age
#    dataset[0][2] = (-1)*dataset[0][2]/max_edu*sd_edu
#    dataset[0][3] = (-1)*dataset[0][3]/max_pe*sd_pe  
#    dataset[0][0] = (-1)*dataset[0][0]+dataset[0][1]*mean_age/sd_age+dataset[0][2]*mean_edu/sd_edu+dataset[0][3]*mean_pe/sd_pe
# 
#    policy_parameters = [float(dataset[0][0]),float(dataset[0][1]),float(dataset[0][2]),float(dataset[0][3]),0,0,0,0,0,0,0,0]

    # Except if EWM is evaluated without age. In that case use instead:
    
#    dataset = pd.read_csv("C:\\Users\\ClaudioSchilter\\Dropbox\\Reinforcement_learning\\Kitagawa_Tetenow_Replication_files\\beta_kt_final_noage_389.csv",header=None)
#    data_directory = os.path.join(local_directory, 'data')
#    df = pd.read_csv(os.path.join(data_directory, "data_lpe2.csv"))
#    #convert policy since KT use different standardization
#    mean_edu = np.mean(df["education"])
#    mean_pe = np.mean(df["prev_earnings"])
#    sd_edu = np.std(df["education"])
#    sd_pe = np.std(df["prev_earnings"])
#    max_edu = np.max(df["education"])
#    max_pe = np.max(df["prev_earnings"])
#    
#    dataset[0][1] = (-1)*dataset[0][1]/max_edu*sd_edu
#    dataset[0][2] = (-1)*dataset[0][2]/max_pe*sd_pe  
#    dataset[0][0] = (-1)*dataset[0][0]+dataset[0][1]*mean_edu/sd_edu+dataset[0][2]*mean_pe/sd_pe
# 
#    policy_parameters = [float(dataset[0][0]),float(dataset[0][1]),float(dataset[0][2])]

    

    # Pseudo random number seed
    set_seed = 24

    # Rewards considered (choose Rlr1 for standard OLS rewards and d_rob_ols_Xfit for doubly robust rewards)
    reward_column_name = "Rlr1"
#    reward_column_name = "d_rob_ols_Xfit"

    # Number of states
    number_of_states_theta = 12 #change to 4 if consider static case (then also change "basis function theta" in part 1 accordingly)
    number_of_states_w = 8

    # Discount factor (use the former for our standard case and the latter for a no-discounting comparison with the EWM policy)
#    beta =  - np.log(0.9)
    beta =  0 # setting discounting to zero for KT comparison

    # Maximum time in years (set to very high number to avoid a binding constraint)
    T = 1

    # Time increments per day
    time_increments_per_day = 100

    # Working days per year
    working_days_per_year = 252

    # Time grid normalisation constant
    time_grid_normalisation = time_increments_per_day*working_days_per_year  # (so currently 25200 increments per year)

    # Lamda normalisation (will be 100)
    lambda_normalisation = int(time_grid_normalisation / working_days_per_year)

    # Normalisation of the rewards (note: rewards are additionally scalled by their standard deviation)
    reward_normalisation = 5309

    # Costs per treatment (normalised by the same constant as the rewards)
    cost_per_treatment = 4/reward_normalisation

    # Total budget
    total_budget = 1

    # Empirical cost of treatment
    empirical_cost_of_treatment = 774


    ##
    ## 2. Data
    ##

    # Creating a logger which tracks the process
    programme = os.path.basename(sys.argv[0])
    log_status = logging.getLogger(programme)
    logging.basicConfig(format='%(asctime)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    log_status.info("running %s" % ' '.join(sys.argv))

    # Input and output folders

    # Creating a unique output folder name
    current_date = datetime.datetime.now()
    current_date = str(current_date)[:-7].replace('-','_').replace(' ','_').replace(':','')


    # Folder paths
    data_directory = os.path.join(local_directory, 'data')
    output_directory = os.path.join(local_directory, f'outcomes')

    # Creating output folder
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loading data

    # RCT data
    df = pd.read_csv(os.path.join(data_directory, "data_lpe2.csv"))

    # Estimated arrival rates
    poisson_table = pd.read_csv(os.path.join(data_directory, "sincos_poisson_means_clusters9.csv"), sep='\t')

    #First dropping NA rows from the relevant rows of the RCT data
    df = df[['age', 'education', 'prev_earnings', reward_column_name, 'clstrs9']]
    df = df.dropna()

    # Scaling covariates
    scaler = StandardScaler()
    scaler.fit(df.loc[:,["age", "education", "prev_earnings"]])
    df.loc[:,["age", "education", "prev_earnings"]] = scaler.transform(df.loc[:,["age", "education", "prev_earnings"]])

    # Subtracting the cost from the reward column
    df[reward_column_name] = df[reward_column_name] - empirical_cost_of_treatment # with 774 being the cost of treatment

    # Preserving the non standardised rewards
    df['rewards_before_standardisation'] = deepcopy(df[reward_column_name])

    # Scaling reward by its standard deviation and an additional normalisation
    df[reward_column_name] = df[reward_column_name]/(reward_normalisation*np.std(df[reward_column_name]))

    # Saving all transformed data to disk for later analysis
    df.to_csv(os.path.join(output_directory, "transformed_rct_data.csv"))

    # Then the RCT data are split and saved in a dictionary with each cluster being a key
    cluster_dict = {}
    for cc in range(0, 4):
        cluster_dict[cc] = df.loc[df.loc[:,'clstrs9'] == (cc+1), ['age', 'education', 'prev_earnings', reward_column_name]]
        cluster_dict[cc].index = range(0, cluster_dict[cc].shape[0])


    # Next the coefficient estimates from the Poisson regression are loaded
    alpha_vector = poisson_table.loc[poisson_table.loc[:,"parm"] == "_cons", "estimate"].values.reshape(4,1)
    beta_vector = poisson_table.loc[poisson_table.loc[:,"parm"] == "dayofyear_sin", "estimate"].values.reshape(4,1)
    gamma_vector = poisson_table.loc[poisson_table.loc[:,"parm"] == "dayofyear_cos", "estimate"].values.reshape(4,1)

    # Lastly, generating the function approximator object which stores all information about the reinforcement learning agents
    function_approximators_object = function_approximators(number_of_states_theta, number_of_states_w)

    ##
    ## 3. Running the evaluation
    ##


    evaluation_returns = eval(EE = EE,
                              policy_parameters = policy_parameters,
                              description = policy_description,
                              sampled_actions = sampled_actions,
                              seed = set_seed,
                              output_directory = output_directory)

    # Can print a reminder what policy exactly was ran
    print("ols_110 with no discounting not sampled")
