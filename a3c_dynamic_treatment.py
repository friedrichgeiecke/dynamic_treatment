
# Importing modules
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.style.use('seaborn')
sns.set_palette('colorblind')
import seaborn as sns
from matplotlib import colors
import numpy as np
import pandas as pd
import os
import math
from math import pi
from random import shuffle
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Process, Array, Value, cpu_count
import time as time_package
import sys
import datetime




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
    def compute_value_parameter_update(self, alpha_w, s_w, I):

        """

        Note:

            - s_theta is the state of the policy function
            - s_w is the state of the value function

            - self.delta_prime is just a float, which is update by the delta prime update function

        """

        # Updating parameter values of the estimator
        return(alpha_w*I*self.delta_prime*s_w[:,0])


    # Compute update for the policy function
    def compute_policy_parameter_update(self, alpha_theta, s_theta, a, I):

        # Updating parameter values of the estimator


        return(alpha_theta*I*self.delta_prime*self.gradient_log_pi(s_theta, a))


##
## Continue rewriting from here before turning to name __main__. Remember to rename s into s_theta etc. in the subprocess funcion etc.
## Once rewritten this function, delete this commend and make commit to repo
##



# This function simulates the entire environment in a sub process
def launch_one_actor_critic_process(mp_w_array,
                                    mp_theta_array,
                                    mp_current_episode,
                                    process_number,
                                    EE,
                                    value_update_only_episodes,
                                    batch_size):


    """

    Calling this function launches a subprocess. There three particular subprocesses that have additional tasks

    # Subprocess 1: runs evaluation episodes with a fixed policy
    # Subprocess 2: keeps track of the current learning episode (process 1 is much slower as it evaluates in between). This is needed to create correct plots.
    # Subprocess 3: saves parameter paths to disk (this also makes it a bit slower which is why it is not used to keep track of time in the other processes)

    """


    # Beginning of the subprocess
    log_status.info(f"Parallel actor-critic process {process_number} launched..")

    # Evaluation cumulative reward array (only used by process 1)
    reward_list_for_evaluation = []
    last_eval = 1 # used to determine the nan elements between evaluation values and updated at every eval
    final_eval_period_flag = False

    # Setting a threat specific seed
    local_state = np.random.RandomState(seed = process_number)

    # Storing the dimension of the value and policy function parameter vectors
    rows_w = len(mp_w_array[:])
    rows_theta = len(mp_theta_array[:])

    # Begin the loop over episodes
    ee = 0
    while ee < EE:


        ##
        ## Training
        ##


        log_status.info(f"Starting episode {ee+1} in process {process_number}")
        tic = time_package.time()

        # Still at an early period only updaing value function?
        if ee < value_update_only_episodes:

            alpha_theta_in_use = 0

        else:

            alpha_theta_in_use = alpha_theta


        # Next, intialise time and budget
        time = 0
        budget_left = deepcopy(total_budget)

        # Initialising I (pre-multiplied by gamme for discounting)
        I = 1

        # People considered and treated in episode
        people_considered_for_treatment = 0
        people_treated = 0


        # Initialise with first person from first cluster
        s0 = cluster_dict[0].loc[0,['age', 'education', 'prev_earnings']].values.reshape(3,1)
        # Building basis functions
        s_theta = np.vstack((1, s0, s0*budget_left, s0*np.cos(2*pi*time), budget_left, np.cos(2*pi*time) ))
        s_w =   np.vstack((budget_left, budget_left*np.sin(2*pi*time), np.cos(2*pi*time)*budget_left,
                           np.cos(2*pi*time)*budget_left**2, budget_left**2, budget_left**3, budget_left**4 ))
        s_R_if_treated = cluster_dict[0].loc[0,reward_column_name]


        # Re-setting the terminal flag to False
        terminal_flag = False

        # Loop over people within episode as long as we are not in a terminal episode
        while terminal_flag == False:

            # Setting the batch update sums to zero
            batch_sum_value_updates = np.zeros([rows_w,])
            batch_sum_policy_updates = np.zeros([rows_theta,])

            # Loop of batch elements
            for bb in range(0, batch_size):

                # Sample action from current policy
                treatment_probability = function_approximators_object.pi(s_theta)
                a = np.random.binomial(1, treatment_probability)

                # Set reward to zero if not treated and otherwise to 1
                if a == 0:

                    R = 0

                # Otherwise update reward, people treated, and budget left
                elif a == 1:

                    R = s_R_if_treated

                    people_treated += 1

                    budget_left -= cost_per_treatment


                # Updating the people count within the episode
                people_considered_for_treatment += 1

                # Draw new waiting time vector lambda
                # -> It is divided by the amount of time units per day (100 in our case). Reason: Lambdas are estimated from daily empirical frequencies,
                # as we have 100 time units per day in our discretisation, this yields the amount of people expected per time increment
                lambda_vector = (np.exp(alpha_vector + (beta_vector * np.sin(time)) + (gamma_vector * np.cos(time))) / lambda_normalisation).reshape(4,1)

                # Summed lambda vector
                summed_lambda_vector = np.sum(lambda_vector)

                # Updating time

                # The lambda fed into an exponential from which now a time increment will be drawn. As lambda is of the scale arrivals per 100th day,
                # the drawn time will be in the same time units. So a drawn value could be 50 as in 50 time units or half a day 50/100. Yet, our time
                # is in years, so we need to transform the drawn time into years. Hence we divide it first by 100 to have days, and then by 252 to have years.
                # This is why time_grid_normalisation has a value of 25200
                delta_time = (math.ceil(np.random.exponential(scale=1/summed_lambda_vector, size=None)) / time_grid_normalisation)
                time = time + delta_time

                # Probably not necessary
                if time > 1:

                    time -= 1

                # Creating discounting factor gamma that is time increment specific
                gamma = np.exp(- beta * delta_time)

                # Drawing the cluster from which the next person arrives
                cluster = np.dot(np.array([0, 1, 2, 3]), np.random.multinomial(1, pvals = (lambda_vector/summed_lambda_vector).reshape(4,)))

                # Drawing the next individual
                sample_person_index = int(np.random.choice(range(0,cluster_dict[cluster].shape[0]), size=1))
                s_prime0 = cluster_dict[cluster].loc[sample_person_index,['age', 'education', 'prev_earnings']].values.reshape(3,1)

                # Building basis functions
                s_theta_prime = np.vstack((1, s_prime0, s_prime0*budget_left, s_prime0*np.cos(2*pi*time), budget_left, np.cos(2*pi*time) ))
                s_w_prime =  np.vstack((budget_left, budget_left*np.sin(2*pi*time), np.cos(2*pi*time)*budget_left,
                                        np.cos(2*pi*time)*budget_left**2, budget_left**2, budget_left**3, budget_left**4 ))
                s_R_if_treated = cluster_dict[cluster].loc[sample_person_index,reward_column_name]

                # Updating the terminal flag
                if budget_left < cost_per_treatment:

                    terminal_flag = True
                    toc = time_package.time()
                    log_status.info(f"Ended episode {ee+1} in process {process_number} which took approximately {(toc-tic):.2f} seconds. {people_considered_for_treatment} people where considered for treatment in this episode, and {people_treated} were treated, until the budget was {budget_left}")

                    # Incrementing the episode by 1
                    ee += 1

                    # Process 2 counts the current learning episodes in the processes other than 1 & 3 (which are slower)
                    if process_number == 2:

                        mp_current_episode.value = mp_current_episode.value + 1

                    break # additionally braking the for loop of the batches if budget ran out

                # Summing up updates for this batch

                # 1. Update TD error
                function_approximators_object.update_delta_prime(R, gamma, s_w, s_w_prime, terminal_flag)

                # 2. Decide whether to update the policy parameters
                theta_eval_index = np.random.binomial(1, theta_eval_prob)
                if theta_eval_index == 1:

                    # Updating cumulative policy updates
                    batch_sum_policy_updates += function_approximators_object.compute_policy_parameter_update(alpha_theta_in_use, s_theta, a, I)

                # 3. Update value function parameters
                batch_sum_value_updates += function_approximators_object.compute_value_parameter_update(alpha_w, s_w, I)

                # Updating the cumulative discount factor I
                I = gamma*I

                # Reassigning state vectors
                s_w = s_w_prime
                s_theta = s_theta_prime

            # After batch is completed, add the cumulative updates to the
            # multi process share parameter objects
            #log_status.info(f"The batch sum value updates are {batch_sum_value_updates}") # for debugging
            mp_w_array[:] = (np.array(mp_w_array[:]) + batch_sum_value_updates).tolist()
            #log_status.info(f"The batch sum policy updates are {batch_sum_policy_updates}") # for debugging
            mp_theta_array[:] = (np.array(mp_theta_array[:]) + batch_sum_policy_updates).tolist()

            # Also for debugging
            #log_status.info(f"The value function parameter array has values {mp_w_array[:]}")
            #log_status.info(f"The policy function parameter array has values {mp_theta_array[:]}")


        # Process 3 saves current parameters for every episode
        if process_number == 3:

            # In episode 1, an array with no rows will be created and as many columns as parameters in the policy function
            if ee == 1:

                policy_parameter_array = np.ones([0, len(function_approximators_object.theta[:])])
                np.savetxt(os.path.join(output_directory, f'policy_parameter_path_{reward_column_name}_rewards.csv'), policy_parameter_array, delimiter=',')


            # In subsequent episodes, this array will be loaded, updated with the current parameter values, and saved again
            else:

                policy_parameter_array = np.loadtxt(os.path.join(output_directory, f'policy_parameter_path_{reward_column_name}_rewards.csv'), delimiter=',')
                if len(policy_parameter_array.shape) == 1:

                    policy_parameter_array = policy_parameter_array.reshape(1, policy_parameter_array.shape[0])

            copy_parameters = deepcopy(function_approximators_object.theta[:])

            policy_parameter_array = np.vstack([policy_parameter_array, np.array(copy_parameters).reshape(1, len(copy_parameters))])
            np.savetxt(os.path.join(output_directory, f'policy_parameter_path_{reward_column_name}_rewards.csv'), policy_parameter_array, delimiter=',')



        # Creating a copy of the current episode of training
        current_ee_representative_process = int(deepcopy(mp_current_episode.value))

        # Process 1 and 3 are slower than the others as they are doing additional tasks (running evaluation episodes, saving policy paths)
        # Hence, their training loops are terminated as soon as the other processes reached EE episodes of training
        if (process_number == 1 or process_number == 3) and current_ee_representative_process == EE:

            break




        ##
        ## Policy evaluation part
        ##


        if process_number == 1 and (ee == 1 or (current_ee_representative_process % evaluation_every_nn_episodes == 0)) and final_eval_period_flag == False:

            if ee == 1:

                current_ee = 1

            else:

                current_ee = deepcopy(current_ee_representative_process)

            # Setting a check to see whether this is the final plot (current_ee because what counts are processes other than 1)
            if current_ee == EE:

                final_eval_period_flag = True


            rewards_these_eval_episodes = []
            rewards_this_eval_episode = 0

            # Saving a local copy of the current policy function parameters
            local_copy_policy_parameters = deepcopy(function_approximators_object.theta[:])


            # Creating a policy function which uses these parameters
            def local_copy_policy_function(s_theta, parameters):

                h = np.dot(s_theta[:,0], parameters)

                treat_prob = 1 - (1/(1 + np.exp(-h)))

                return treat_prob


            log_status.info(f"Evaluation episode in process {process_number} started after {current_ee} episodes of training in the parallel actor critic model.")

            for evaluation_episode in range(0, evaluation_episodes):

                rewards_this_eval_episode = 0

                tic = time_package.time()

                # Next, intialise time and budget
                time = 0
                budget_left = deepcopy(total_budget)

                # Initialising I (pre-multiplied by gamme for discounting)
                I = 1

                # People considered and treated in episode
                people_considered_for_treatment = 0
                people_treated = 0


                # Initialise with first person from first cluster
                s0 = cluster_dict[0].loc[0,['age', 'education', 'prev_earnings']].values.reshape(3,1)
                # Adding interactions
                s_theta = np.vstack((1, s0, s0*budget_left, s0*np.cos(2*pi*time), budget_left, np.cos(2*pi*time) ))
                s_w =  np.vstack((budget_left, budget_left*np.sin(2*pi*time), np.cos(2*pi*time)*budget_left,
                                  np.cos(2*pi*time)*budget_left**2, budget_left**2, budget_left**3, budget_left**4 ))
                s_R_if_treated = cluster_dict[0].loc[0,reward_column_name]
                #s_compprob = cluster_dict[0].loc[0,'compProp']


                # Re-setting the terminal flag to False
                terminal_flag = False

                # Loop over people within episode as long as we are not in a terminal episode
                while terminal_flag == False:

                    for bb in range(0, batch_size):

                        if ee == 1:

                            treatment_probability = 0.5

                        else:

                            treatment_probability = local_copy_policy_function(s_theta, local_copy_policy_parameters)

                        a = np.random.binomial(1, treatment_probability)

                        if a == 0:

                            R = 0

                        elif a == 1 and budget_left > cost_per_treatment:

                            R = s_R_if_treated

                            people_treated += 1

                            budget_left -= cost_per_treatment


                        # Collecting the total episode reward for evaluation
                        rewards_this_eval_episode += I*R


                        # Updating the people count within the episode
                        people_considered_for_treatment += 1


                        # Draw new waiting time vector lambda
                        lambda_vector = (np.exp(alpha_vector + (beta_vector * np.sin(time)) + (gamma_vector * np.cos(time))) / lambda_normalisation).reshape(4,1)

                        # Summed lambda vector
                        summed_lambda_vector = np.sum(lambda_vector)

                        # Updating time
                        delta_time = (math.ceil(np.random.exponential(scale=1/summed_lambda_vector, size=None))/time_grid_normalisation)
                        time = time + delta_time

                        if time > 1:

                            time -= 1

                        # Creating a period specific discounting factor gamma
                        gamma = np.exp(-beta * delta_time)


                        # Obtaining from which cluster the next person arrives
                        cluster = np.dot(np.array([0, 1, 2, 3]), np.random.multinomial(1, pvals = (lambda_vector/summed_lambda_vector).reshape(4,)))


                        # Obtaining the next state
                        sample_person_index = int(np.random.choice(range(0,cluster_dict[cluster].shape[0]), size=1))
                        s_prime0 = cluster_dict[cluster].loc[sample_person_index,['age', 'education', 'prev_earnings']].values.reshape(3,1)

                        # Adding interactions
                        s_theta_prime = np.vstack((1, s_prime0, s_prime0*budget_left, s_prime0*np.cos(2*pi*time), budget_left, np.cos(2*pi*time) ))
                        s_w_prime =  np.vstack((budget_left, budget_left*np.sin(2*pi*time), np.cos(2*pi*time)*budget_left,
                                                np.cos(2*pi*time)*budget_left**2, budget_left**2, budget_left**3, budget_left**4 ))
                        s_R_if_treated = cluster_dict[cluster].loc[sample_person_index,reward_column_name]
                        #s_compprob = cluster_dict[cluster].loc[sample_person_index,'compProp']


                        # Updating the terminal flag
                        if budget_left < cost_per_treatment:

                            terminal_flag = True
                            toc = time_package.time()
                            log_status.info("Current evaluation ee is: {current_ee}. Ended evaluation episode {evaluation_episode+1} in process {process_number} which took approximately {toc-tic:.2f} seconds. {people_considered_for_treatment} people where considered for treatment in this episode, and {people_treated} were treated, until the budget was {budget_left}")

                            break # additionally braking the for loop for the batches
                            #log_status.info(time)


                        # Updating I
                        I = gamma*I

                        # Reassigning state vectors
                        s_w = s_w_prime
                        s_theta = s_theta_prime


                rewards_these_eval_episodes.append(rewards_this_eval_episode)


            # For the very first evaluation, average rewards are added that have been precomputed as averages over many episodes
            # Unhash and change evaluation_every_nn_episodes to very high number when determinin random policy benchmark
            #if current_ee == 1:
            #
            #    np.savetxt('random_policy_reward_mean_{}_over_{}_episodes.csv'.format(reward_column_name, evaluation_every_nn_episodes), np.array([np.mean(rewards_these_eval_episodes)]), delimiter=',')


            if current_ee == 1:

                if reward_column_name == "doubly_robust_ols":

                    # Value obtained as an average of 2000 evaluation episodes with 0.5/0.5 policy
                    reward_mean_these_eval_episodes = 0.0361874678946277

                if reward_column_name == "d_rob_ols_Xfit":

                    # Value obtained as an average of 10,000 evaluation episodes with 0.5/0.5 policy
                    reward_mean_these_eval_episodes = 0.0216010850940042


                if reward_column_name == "Rlr1":

                    # Value obtained as an average of 500 evaluation episodes with 0.5/0.5 policy
                    reward_mean_these_eval_episodes = 1.21061062127504

            # For subsequent evaluation episodes, average rewards are computed over a list of length 'evaluation_episodes'
            elif current_ee > 1:

                reward_list_for_evaluation = reward_list_for_evaluation + [None] * (int(current_ee) - last_eval - 1)
                reward_mean_these_eval_episodes = np.mean(rewards_these_eval_episodes)

            last_eval = deepcopy(int(current_ee))
            reward_list_for_evaluation.append(reward_mean_these_eval_episodes)
            x_axis_range = list(range(1, (len(reward_list_for_evaluation)+1)))

            # Printing x and y
            #log_status.info(x_axis_range)
            #log_status.info(reward_list_for_evaluation)

            x_array = np.array(x_axis_range).astype(np.double)
            y_array = np.array(reward_list_for_evaluation).astype(np.double)
            y_mask = np.isfinite(y_array)

            # Saving reward arrays to disk (non-normalised)
            np.savetxt('x_axis_for_rewards_{}.csv'.format(reward_column_name), x_array, delimiter=',')
            np.savetxt('y_axis_for_rewards_{}.csv'.format(reward_column_name), y_array, delimiter=',')

            # Normalising current reward arrays for plotting
            y_array = y_array / y_array[0]

            # Plotting the figure
            plt.figure(figsize=(12,4))
            plt.plot(x_array[y_mask], y_array[y_mask], linestyle='--', marker='o')
            plt.title('Reward trajectory')
            plt.ylabel('Average cumulative episode reward achieved')
            plt.xlabel('Episodes approximately trained in each of {n_parallel_processes} parallel threads')
            plt.savefig(os.path.join(data_directory, f'current_reward_trajectory_{reward_column_name}_rewards.pdf'), dpi = 300) # saved into data directory
            # plt.show() # muted on fabian

            # Saving an extra plot after 1000's of episodes
            if int(current_ee) % 10000 == 0:

                plt.savefig(os.path.join(data_directory, f'reward_trajectory_{reward_column_name}_rewards_after_{int(current_ee)}_episodes.pdf'), dpi = 300) # save extra plot every x-1000 episodes





if __name__ == "__main__":

    # To do:
    #
    # 1) Even with a lot of evaluation periods, the random policy initial period reward seems quite different. Maybe recompute this with 20k eval episodes to see whether it is the same.


    # Creating a logger which tracks the process
    programme = os.path.basename(sys.argv[0])
    log_status = logging.getLogger(programme)
    logging.basicConfig(format='%(asctime)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    log_status.info("running %s" % ' '.join(sys.argv))


    # Setting a seed
    np.random.seed(seed=42)

    # Input and output folders
    current_date = datetime.datetime.now()
    current_date = str(current_date)[:-7].replace('-','_').replace(' ','_').replace(':','_')
    data_directory = '/users/geieckef/dynamic_treatment/data/'
    output_directory = f'/users/geieckef/dynamic_treatment/run_{current_date}/'

    # Loading data

    # RCT data
    df = pd.read_csv(os.path.join(data_directory, "data_lpe2.csv"))

    # Estimated arrival rates
    poisson_table = pd.read_csv(os.path.join(data_directory, "sincos_poisson_means_clusters9.csv"), sep='\t')


# Continue rewriting from here. Set all hyper-parameters etc and write them into a txt file in the folder of the current run.
# Also add a line to this txt file with the spec of the value and of the polciy function (needs to be changed manually whenever these are changed in the code later)
# Alternatively, build a transformation function that is updated and call it also here






os.chdir("/users/geieckef/dynamic_treatment/data/")
df = pd.read_csv("data_lpe2.csv")

#the two versions for the first draft (ols with just our 3 variables, once doubly robust, once not):

reward_option = int(sys.argv[1])

# Plain doubly robust OLS rewards option
if reward_option == 1:

    reward_column_name = "doubly_robust_ols"
    alpha_theta = 0.3
    alpha_w = 0.8
    EE = 20000
    evaluation_every_nn_episodes = 100
    evaluation_episodes = 100 # works because 100 evaluation episodes take less time than 100 training episodes because no function approximator updates
    batch_size = 128

    # Costs per treatment
    cost_per_treatment = 1/6400
    # Total budget
    total_budget = 0.25


# Doubly robust OLS rewards option crossfitted
if reward_option == 2:

    reward_column_name = "d_rob_ols_Xfit"
    alpha_theta = 0.1 # 0.075/0.2 workds with EE = 100,000 and batch_size = 128
    alpha_w = 0.2 #
    EE = 100000
    evaluation_every_nn_episodes = 500 # change both to 500 when once random reward number saved
    evaluation_episodes = 500 # works because 100 evaluation episodes take less time than 100 training episodes because no function approximator updates
    batch_size = 1024

    # Costs per treatment
    cost_per_treatment = (1/6400) * 2 # factor was 1 previously
    # Total budget
    total_budget = 0.25 * 2 # factor was 1 previously


# Normal OLS rewards option
elif reward_option == 3:

    reward_column_name = "Rlr1"
    alpha_theta = 0.3 # 0.8/10 worked well before recent changes in arrival etc
    alpha_w = 5
    EE = 20000
    evaluation_every_nn_episodes = 100
    evaluation_episodes = 5
    batch_size = 128

    # Costs per treatment
    cost_per_treatment = 1/6400
    # Total budget
    total_budget = 0.25

# First dropping NA rows from the relevant rows of the dataframe
#df = df[['age', 'education', 'prev_earnings', reward_column_name, 'clstrs9', 'compProp']]
df = df[['age', 'education', 'prev_earnings', reward_column_name, 'clstrs9']]
df = df.dropna()


df[reward_column_name] = df[reward_column_name] - 774 # with 774 being the cost of treatment


# Loading the poisson data
poisson_table = pd.read_csv("sincos_poisson_means_clusters9.csv", sep='\t')
poisson_table

alpha_vector = poisson_table.loc[poisson_table.loc[:,"parm"] == "_cons", "estimate"].values.reshape(4,1)
beta_vector = poisson_table.loc[poisson_table.loc[:,"parm"] == "dayofyear_sin", "estimate"].values.reshape(4,1)
gamma_vector = poisson_table.loc[poisson_table.loc[:,"parm"] == "dayofyear_cos", "estimate"].values.reshape(4,1)


# In[59]:


# Alternatively normalise the covariates and also the reward
scaler = StandardScaler()
scaler.fit(df.loc[:,["age", "education", "prev_earnings"]])
df.loc[:,["age", "education", "prev_earnings"]] = scaler.transform(df.loc[:,["age", "education", "prev_earnings"]])
df[reward_column_name] = df[reward_column_name]/(1000*np.std(df[reward_column_name]))
log_status.info(np.mean(df["prev_earnings"]))
log_status.info(np.std(df["prev_earnings"]))
log_status.info(np.mean(df[reward_column_name]))
log_status.info(np.std(df[reward_column_name]))


# In[39]:


cluster_dict = {}
for cc in range(0, 4):
    #cluster_dict[cc] = df.loc[df.loc[:,'clstrs9'] == (cc+1), ['age', 'education', 'prev_earnings', reward_column_name, 'compProp']]
    cluster_dict[cc] = df.loc[df.loc[:,'clstrs9'] == (cc+1), ['age', 'education', 'prev_earnings', reward_column_name]]
    cluster_dict[cc].index = range(0, cluster_dict[cc].shape[0])


# In[40]:


# Number of learning episodes
#EE = 5000
#EE = 100

# Number of actions
number_of_actions = 2

# Number of states
number_of_states = 12
number_of_states_w = 7

# Learning rate
#alpha_theta = 0.01 # 0.8/10 worked well
#alpha_theta = 0
#alpha_w = 0.3 # 10 worked well

# For the old rewards rlr1 0.8/10 and 2/20 worked well



# Periods in the beginning where only the value function is updated
value_update_only_episodes = 0 # 10

# Discount factor
beta =  - np.log(0.9)

#number of steps in which theta is updated
theta_eval_prob = 0.99

# Working days per year
working_days_per_year = 252
#working_days_per_year = 248

# Time grid normalisation constant
time_grid_normalisation = 25200 # (so 252 working days each with 100 time intervals)
#time_grid_normalisation = 24800

# Lamda normalisation
lambda_normalisation = int(time_grid_normalisation / working_days_per_year)



# Creating the policy estimator object. This now needs to be created at the beginning of every subprocess
#function_approximators_object = function_approximators(number_of_states, number_of_actions)


# ### Defining the function for one subprocess

# In[56]:





# Generating the function approximator object
function_approximators_object = function_approximators(number_of_states, number_of_states_w, number_of_actions)

# Define the mp arrays

# For the two parameter vectors
mp_w_array = Array('d', ([0] * number_of_states_w)) # d refers to double precision float
mp_theta_array = Array('d', ([0] * (number_of_states*(number_of_actions-1))))
# For the episode reached in a non-evaluation process
mp_current_episode = Value('d', 0.0)

# Setting the function approximator parameters equal to these arrays
function_approximators_object.w = mp_w_array
function_approximators_object.theta = mp_theta_array

n_parallel_processes = max(1, cpu_count() - 1)
#n_parallel_processes = 1


# Process list
processes = [Process(target=launch_one_actor_critic_process, args=(mp_w_array,
                                                               mp_theta_array,
                                                               mp_current_episode,
                                                               process_number+1,
                                                               EE,
                                                               value_update_only_episodes,
                                                               batch_size)) for process_number in range(0, n_parallel_processes)]

# Run processes

seconds_in_between_process_starts = 1

for p in processes:

    # put sleep timer here, probably best random mean around a third of an episode time and decent sigma
    time_package.sleep(seconds_in_between_process_starts)
    p.start()



# Exit the completed processes
for p in processes:
    p.join()






#
#
# Question from past runs: Why does the doubly robust estimate evaluate after 90 periods rather than 100? The plain OLS evaluates after exaclty 100 episodes.
#
#
