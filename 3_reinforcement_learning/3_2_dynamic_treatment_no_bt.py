# Importing modules
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend("agg")
plt.style.use("seaborn")
import numpy as np
import pandas as pd
import os
import math
from math import pi
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from multiprocessing import Process, Array, Value, cpu_count
import time as time_package
import sys
import logging
import datetime
import inspect
import gc


# Reinforcement learning agent class
class ACAgent:

    # Initialise parameters and weights
    def __init__(self, number_of_states_theta, number_of_states_w):

        self.number_of_states_theta = number_of_states_theta
        self.number_of_states_w = number_of_states_w
        self.delta_prime = 0
        self.theta = [0] * (self.number_of_states_w)
        self.w = [0] * self.number_of_states_w

    # Policy basis function
    def return_basis_function_theta(self, s0, budget_left, time, T):

        s_theta = np.vstack(
            (
                1,
                s0,
            )
        )

        return s_theta

    # Value basis function
    def return_basis_function_w(self, budget_left, time, T):

        s_w = np.vstack(
            (
                (budget_left) * (T - time),
                (budget_left) * (T - time) ** 2,
                (budget_left) ** 2 * (T - time),
                ((budget_left) * (T - time)) ** 2,
                (budget_left) * np.sin(pi * time),
                (budget_left) * np.sin(2 * pi * time),
                (budget_left) ** 2 * np.sin(pi * time),
                ((budget_left) ** 2 * np.sin(2 * pi * time)),
                (budget_left) ** 3 * (T - time),
            )
        )

        return s_w

    # Policy function
    def pi(self, s_theta):

        h = np.dot(s_theta[:, 0], self.theta[:])

        treat_prob = 1 - (1 / (1 + np.exp(-h)))

        return treat_prob

    # Value function
    def v(self, s_w):

        output = np.dot(s_w[:, 0], self.w[:])

        return output

    # Gradient of the log policy
    def gradient_log_pi(self, s_theta, a):

        output = (-(a - self.pi(s_theta)) * s_theta)[:, 0]

        return output

    # TD error
    def update_delta_prime(self, R, gamma, s_w, s_prime_w, terminal_flag):

        if terminal_flag == False:

            self.delta_prime = R + gamma * self.v(s_prime_w) - self.v(s_w)

        else:

            self.delta_prime = 0

    # Return the update for the value function parameters
    def value_parameter_update(self, alpha_w, s_w):

        return alpha_w * self.delta_prime * s_w[:, 0]

    # Return the update for the policy function parameters
    def policy_parameter_update(self, alpha_theta, s_theta, a, I):

        return alpha_theta * I * self.delta_prime * self.gradient_log_pi(s_theta, a)


# Subprocess launching function
def launch_one_actor_critic_process(
    mp_w_array, mp_theta_array, mp_current_episode, process_number,
):

    """


    Calling this function launches a subprocess. Training happens in all processes,
    however, there are three particular subprocesses which have additional tasks

    Subprocess 1: Saves the current parameters to disks each episode
    Subprocess 2: Runs evaluations evaluation episodes with a fixed policy
    Subprocess 3: Keeps track of the approximate learning episode in most processes
                  (this is not done in process 1 or 2 as these are slower than the rest)

    """

    # Beginning of the subprocess
    log_status.info(f"Parallel actor-critic process {process_number} launched..")

    # Create a local agent instance
    agent = ACAgent(number_of_states_theta, number_of_states_w)

    # Set its parameters equal to the global MP arrays to access them at all times
    agent.w = mp_w_array
    agent.theta = mp_theta_array

    # Evaluation variables (only used by process 2)
    reward_list_for_evaluation = []
    last_eval = 1  # episode in which the last evaluation happened
    last_plot = 1  # episode in which the last plot was created
    final_eval_period_flag = False

    # Set a thread specific seed
    local_state = np.random.RandomState(seed=process_number)

    # Begin the loop over episodes
    ee = 0
    while ee < EE:

        ##############################################################################
        ##
        ## Training (all processes)
        ##
        ##############################################################################

        log_status.info(f"Starting episode {ee+1} in process {process_number}")
        tic = time_package.time()

        # Set the policy learning rate to zero as long as only value function is updated
        if ee < value_update_only_episodes:

            alpha_theta_in_use = 0

        else:

            alpha_theta_in_use = alpha_theta

        # Initialisations
        time = 0
        budget_left = deepcopy(total_budget)
        I = 1
        people_considered_for_treatment = 0
        people_treated = 0

        # Determine the first person which will arrive at the agent (always the first
        # person from the first cluster)

        # Covariates (state before adding transformations)
        s0 = (
            cluster_dict[0]
            .loc[0, ["age", "education", "prev_earnings"]]
            .values.reshape(3, 1)
        )

        # Build basis functions
        s_theta = agent.return_basis_function_theta(
            s0=s0, budget_left=budget_left, time=time, T=T
        )
        s_w = agent.return_basis_function_w(budget_left=budget_left, time=time, T=T)

        # Treatment reward
        s_R_if_treated = cluster_dict[0].loc[0, reward_column_name]

        # Reset the terminal flag to False (it signals if budget or time are exhausted)
        terminal_flag = False

        # Loop over people within episode as long as budget and time allow
        while terminal_flag == False:

            # Set the batch update sums to zero
            batch_sum_value_updates = np.zeros([agent.number_of_states_w,])
            batch_sum_policy_updates = np.zeros([agent.number_of_states_theta,])

            # Loop over batch elements
            for bb in range(0, batch_size):

                # Sample action from current policy
                treatment_probability = agent.pi(s_theta)
                a = np.random.binomial(1, treatment_probability)

                # Set reward to zero if not treated
                if a == 0:

                    R = 0

                # Otherwise obtain reward, update people treated, and budget left
                elif a == 1:

                    R = s_R_if_treated

                    people_treated += 1

                    budget_left -= cost_per_treatment

                # Update the people count within the episode
                people_considered_for_treatment += 1

                # For each cluster, draw the expected number of people arriving per
                # within day time increment (we e.g. set 100 time increments per day)
                lambda_vector = (
                    np.exp(
                        alpha_vector
                        + (beta_vector * np.sin(time))
                        + (gamma_vector * np.cos(time))
                    )
                    / time_increments_per_day
                ).reshape(4, 1)

                # Obtain the expected number of people (irrespective of clusters)
                # arriving per within day time increment
                summed_lambda_vector = np.sum(lambda_vector)

                # Draw the time increment until the next arrival and normalise it
                # to a fraction of yearly time
                delta_time = math.ceil(
                    np.random.exponential(scale=1 / summed_lambda_vector, size=None)
                ) / (time_increments_per_day * working_days_per_year)
                time = time + delta_time

                # Compute the discounting factor for the next arrival
                gamma = np.exp(-beta * delta_time)

                # Draw the cluster from which the next person arrives
                cluster = np.dot(
                    np.array([0, 1, 2, 3]),
                    np.random.multinomial(
                        1, pvals=(lambda_vector / summed_lambda_vector).reshape(4,)
                    ),
                )

                # Draw the next individual
                sample_person_index = int(
                    np.random.choice(range(0, cluster_dict[cluster].shape[0]), size=1)
                )
                s0_prime = (
                    cluster_dict[cluster]
                    .loc[sample_person_index, ["age", "education", "prev_earnings"]]
                    .values.reshape(3, 1)
                )

                # Build basis functions
                s_theta_prime = agent.return_basis_function_theta(
                    s0=s0_prime, budget_left=budget_left, time=time, T=T
                )
                s_w_prime = agent.return_basis_function_w(
                    budget_left=budget_left, time=time, T=T
                )

                # Treatment reward
                s_R_if_treated = cluster_dict[cluster].loc[
                    sample_person_index, reward_column_name
                ]

                # Update the terminal flag if episode finishes
                if budget_left < cost_per_treatment or time >= T:

                    terminal_flag = True
                    toc = time_package.time()
                    log_status.info(
                        f"Ended episode {ee+1} in process {process_number} which took approximately {(toc-tic):.2f} seconds. {people_considered_for_treatment} people where considered for treatment in this episode, {people_treated} were treated, until the budget was {budget_left:.4f}, and time was {time:.4f}."
                    )

                    # Increment the episode count by 1
                    ee += 1

                    # Process 3 counts and approximates the current learning
                    # episodes in the processes other than 1 & 2 (which are slower)
                    if process_number == 3:

                        mp_current_episode.value = mp_current_episode.value + 1

                    # Additionally braking the for loop of the batches if budget ran out
                    break

                # Adding the current period to the batch

                # Update TD error
                agent.update_delta_prime(R, gamma, s_w, s_w_prime, terminal_flag)

                # Decide whether to update the policy parameters
                theta_eval_index = np.random.binomial(1, theta_eval_prob)
                if theta_eval_index == 1:

                    # Policy batch
                    batch_sum_policy_updates += agent.policy_parameter_update(
                        alpha_theta_in_use, s_theta, a, I
                    )

                # Value batch
                batch_sum_value_updates += agent.value_parameter_update(alpha_w, s_w)

                # Update the cumulative discount factor I
                I = gamma * I

                # Reassign state vectors
                s_w = s_w_prime
                s_theta = s_theta_prime

            # After the batch is completed, add the cumulative updates to the MP arrays
            # which are accessed across processes
            mp_w_array[:] = (np.array(mp_w_array[:]) + batch_sum_value_updates).tolist()
            mp_theta_array[:] = (
                np.array(mp_theta_array[:]) + batch_sum_policy_updates
            ).tolist()

        # Resetting a couple of key assignments after the episode, to note should any
        # be wrongly reused in evaluation below or the next episode as most variables
        s0 = None
        s0_prime = None
        s_w = None
        s_w_prime = None
        s_theta = None
        s_theta_prime = None
        s_R_if_treated = None

        ##############################################################################
        ##
        ## Saving parameter paths (only process 1)
        ##
        ##############################################################################

        # At the end of an episode, process 1 additionally saves current parameters
        if process_number == 1:

            # Creating the files in the first episode
            if ee == 1:

                value_file_path = os.path.join(
                    output_directory, "value_parameter_path.csv"
                )
                policy_file_path = os.path.join(
                    output_directory, "policy_parameter_path.csv"
                )

                value_file = open(value_file_path, "w", buffering = 10)
                policy_file = open(policy_file_path, "w", buffering = 10)

                value_header = "episode, " + ", ".join(
                    [
                        f"coefficient_{coeff_count_value}"
                        for coeff_count_value in range(agent.number_of_states_w)
                    ]
                )
                policy_header = "episode, " + ", ".join(
                    [
                        f"coefficient_{coeff_count_policy}"
                        for coeff_count_policy in range(agent.number_of_states_theta)
                    ]
                )

                # Headers
                value_file.write(value_header + "\n")
                policy_file.write(policy_header + "\n")

                # Starting values
                value_file.write(
                    ", ".join(["0" for _ in range(agent.number_of_states_w + 1)]) + "\n"
                )
                policy_file.write(
                    ", ".join(["0" for _ in range(agent.number_of_states_theta + 1)])
                    + "\n"
                )

            # Writing parameters into the paths otherwise
            else:

                copy_value_parameters = deepcopy(mp_w_array[:])
                copy_policy_parameters = deepcopy(mp_theta_array[:])
                copy_current_episode = int(deepcopy(mp_current_episode.value))

                value_line_item = f"{copy_current_episode} , " + ", ".join(
                    [str(coeff_value) for coeff_value in copy_value_parameters]
                )
                policy_line_item = f"{copy_current_episode} , " + ", ".join(
                    [str(coeff_policy) for coeff_policy in copy_policy_parameters]
                )

                value_file.write(value_line_item + "\n")
                policy_file.write(policy_line_item + "\n")

                # Trying to flush files regularly to store progress on disk and free
                # memory
                if ee % 500 == 0:

                    value_file.flush()
                    policy_file.flush()
                    gc.collect()


        ##############################################################################
        ##
        ## Policy evaluation (only process 2)
        ##
        ##############################################################################

        # If in process 2 and either a) in the very first episode or
        # b) not evaluated for at least evaluation_every_nn_episodes -> start evaluation
        if (
            process_number == 2
            and (
                ee == 1
                or (
                    (int(mp_current_episode.value) - last_eval)
                    >= evaluation_every_nn_episodes
                )
            )
            and final_eval_period_flag == False
        ):

            # First, determine the current global evaluation episode (rather than the
            # local ee of process 2 which will lag behind)
            if ee == 1:

                current_evaluation_episode = 1

            else:

                current_evaluation_episode = deepcopy(int(mp_current_episode.value))

            # Fix agent parameters at current global parameters
            agent.w = deepcopy(mp_w_array[:])
            agent.theta = deepcopy(mp_theta_array[:])

            # Status update
            log_status.info(
                f"Evaluation episode in process {process_number} started after {current_evaluation_episode} episodes of training in the parallel actor critic model."
            )

            # Setting a flag whether this is the final plot
            if current_evaluation_episode == EE:

                final_eval_period_flag = True

            # Starting the evaluation loop
            rewards_these_eval_episodes = []
            for evaluation_episode in range(0, evaluation_episodes):

                # Initialisations
                tic = time_package.time()
                rewards_this_eval_episode = 0
                time = 0
                budget_left = deepcopy(total_budget)
                I = 1
                people_considered_for_treatment = 0
                people_treated = 0

                # Determine the first person which will arrive at the agent (always the
                # first person from the first cluster)

                # Covariates (state before adding transformations)
                s0 = (
                    cluster_dict[0]
                    .loc[0, ["age", "education", "prev_earnings"]]
                    .values.reshape(3, 1)
                )

                # Build basis functions
                s_theta = agent.return_basis_function_theta(
                    s0=s0, budget_left=budget_left, time=time, T=T
                )
                s_w = agent.return_basis_function_w(
                    budget_left=budget_left, time=time, T=T
                )

                # Treatment reward
                s_R_if_treated = cluster_dict[0].loc[0, reward_column_name]

                # Reset the terminal flag to False
                # (it signals if budget or time are exhausted)
                terminal_flag = False

                # Loop over people within episode until budget or time exhausted
                while terminal_flag == False:

                    # In first evaluation episode actions are 50/50
                    if ee == 1:

                        treatment_probability = 0.5

                    # Otherwise use the locally saved policy
                    else:

                        treatment_probability = agent.pi(s_theta)

                    # Sample action from fixed evaluation policy
                    a = np.random.binomial(1, treatment_probability)

                    # Set reward to zero if not treated
                    if a == 0:

                        R = 0

                    # Otherwise obtain reward, update people treated, and budget
                    # left
                    elif a == 1 and budget_left > cost_per_treatment:

                        R = s_R_if_treated

                        people_treated += 1

                        budget_left -= cost_per_treatment

                    # Collecting the total episode reward for evaluation
                    rewards_this_eval_episode += I * R

                    # Update the people count within the episode
                    people_considered_for_treatment += 1

                    # For each cluster, draw the expected number of people arriving per
                    # within day time increment (we e.g. set 100 time increments per day)
                    lambda_vector = (
                        np.exp(
                            alpha_vector
                            + (beta_vector * np.sin(time))
                            + (gamma_vector * np.cos(time))
                        )
                        / time_increments_per_day
                    ).reshape(4, 1)

                    # Obtain the expected number of people (irrespective of clusters)
                    # arriving per within day time increment
                    summed_lambda_vector = np.sum(lambda_vector)

                    # Draw the time increment until the next arrival and normalise it
                    # to a fraction of yearly time
                    delta_time = math.ceil(
                        np.random.exponential(scale=1 / summed_lambda_vector, size=None)
                    ) / (time_increments_per_day * working_days_per_year)
                    time = time + delta_time

                    # Compute the discounting factor for the next arrival
                    gamma = np.exp(-beta * delta_time)

                    # Draw the cluster from which the next person arrives
                    cluster = np.dot(
                        np.array([0, 1, 2, 3]),
                        np.random.multinomial(
                            1, pvals=(lambda_vector / summed_lambda_vector).reshape(4,),
                        ),
                    )

                    # Draw the next individual
                    sample_person_index = int(
                        np.random.choice(
                            range(0, cluster_dict[cluster].shape[0]), size=1
                        )
                    )
                    s0_prime = (
                        cluster_dict[cluster]
                        .loc[
                            sample_person_index, ["age", "education", "prev_earnings"],
                        ]
                        .values.reshape(3, 1)
                    )

                    # Build basis functions
                    s_theta_prime = agent.return_basis_function_theta(
                        s0=s0_prime, budget_left=budget_left, time=time, T=T
                    )
                    s_w_prime = agent.return_basis_function_w(
                        budget_left=budget_left, time=time, T=T
                    )

                    # Treatment reward
                    s_R_if_treated = cluster_dict[cluster].loc[
                        sample_person_index, reward_column_name
                    ]

                    # Update the terminal flag if episode finishes
                    if budget_left < cost_per_treatment or time >= T:

                        terminal_flag = True
                        toc = time_package.time()
                        log_status.info(
                            f"Currently evaluating policy after {current_evaluation_episode} episodes of training: Ended evaluation episode {evaluation_episode+1} in process {process_number} which took approximately {toc-tic:.2f} seconds. {people_considered_for_treatment} people where considered for treatment in this episode, {people_treated} were treated, until the budget was {budget_left:.4f}, and time was {time:.4f}. The cumulative reward achieved in the episode was {np.sum(rewards_this_eval_episode):.6f}."
                        )

                        break  # additionally breaking the for loop for the batches

                    # Update the cumulative discount factor I
                    I = gamma * I

                    # Reassign state vectors
                    s_w = s_w_prime
                    s_theta = s_theta_prime

                # Once the episode is finished, append the discounted reward
                rewards_these_eval_episodes.append(rewards_this_eval_episode)

            # In the first evaluation episode, overwrite the average reward with a
            # random reward obtained from evaluation over many more episodes
            if current_evaluation_episode == 1:

                if reward_column_name == "Rlr1":

                    # Value obtained as an average of 100,000 evaluation episodes with
                    # 0.5/0.5 policy
                    reward_mean_these_eval_episodes = 0.18993510953729537

                if reward_column_name == "d_rob_ols_Xfit":

                    # Value obtained as an average of 100,000 evaluation episodes with
                    # 0.5/0.5 policy
                    reward_mean_these_eval_episodes = 0.0033400270414006734

            # For subsequent evaluations episodes, add nan values for those episodes
            # in which the functions have not been evaluated
            else:

                reward_list_for_evaluation = reward_list_for_evaluation + [None] * (
                    int(current_evaluation_episode) - last_eval - 1
                )
                reward_mean_these_eval_episodes = np.mean(rewards_these_eval_episodes)

            # Update the last_eval value
            last_eval = deepcopy(int(current_evaluation_episode))

            # Create reward arrays for plot
            reward_list_for_evaluation.append(reward_mean_these_eval_episodes)
            x_axis_range = list(range(1, (len(reward_list_for_evaluation) + 1)))
            x_array = np.array(x_axis_range).astype(np.double)
            y_array = np.array(reward_list_for_evaluation).astype(np.double)
            y_mask = np.isfinite(y_array)

            # Save reward arrays to disk (not normalised by random reward which is
            # stored in the beginning)
            np.savetxt(
                os.path.join(
                    output_directory, f"x_axis_for_rewards_{reward_column_name}.csv"
                ),
                x_array,
                delimiter=",",
            )
            np.savetxt(
                os.path.join(
                    output_directory, f"y_axis_for_rewards_{reward_column_name}.csv"
                ),
                y_array,
                delimiter=",",
            )

            # Normalise current reward arrays for plotting
            y_array = y_array / y_array[0]

            # Set the colour palette
            colour_palette_in_use = sns.color_palette("husl", 12)

            # Plot the figure
            fig = plt.figure(figsize=(12, 4))
            plt.plot(
                x_array[y_mask],
                y_array[y_mask],
                linestyle="--",
                marker="o",
                color=colour_palette_in_use[8],
            )
            plt.title("\nWelfare trajectory\n")
            plt.ylabel("Average cumulative episode welfare achieved\n")
            plt.xlabel(
                f"\nEpisodes approximately trained in each of {n_parallel_processes_selected} parallel processes"
            )
            fig.tight_layout()
            plt.savefig(
                os.path.join(
                    output_directory,
                    f"current_reward_trajectory_{reward_column_name}_rewards.pdf",
                ),
                dpi=300,
            )

            # Save a backup plot each few thousand observations
            if int(current_evaluation_episode - last_plot) > 1000:

                plt.savefig(
                    os.path.join(
                        backup_directory,
                        f"reward_trajectory_{reward_column_name}_rewards_after_approx_{int(current_evaluation_episode)}_episodes.pdf",
                    ),
                    dpi=300,
                )  # save extra plot every x-1000 episodes
                last_plot = deepcopy(current_evaluation_episode)

            # At the end of the evaluation episode, set parameters in process 2 equal to
            # the global MP arrays again
            agent.w = mp_w_array
            agent.theta = mp_theta_array

        # Process 1 and 2 are slower than the others as they are doing additional tasks
        # (saving policy paths, running evaluation episodes) Hence, their training loops
        # are terminated as soon as the other processes reached EE episodes of training
        if (process_number == 1 or process_number == 2) and int(
            mp_current_episode.value
        ) == EE:

            if process_number == 1:

                value_file.close()
                policy_file.close()

            break


if __name__ == "__main__":



    ###################################################################################
    # Parameter settings start
    ###################################################################################


    ##
    ## 1. Common parameters
    ##

    # Number of states of the policy basis function (value basis chosen as part of grid)
    number_of_states_theta = 4

    # Periods in the beginning where only the value function is updated
    value_update_only_episodes = 0

    # Discount factor
    beta = -np.log(0.9)

    # Probability with which the policy function is updated
    theta_eval_prob = 1

    # Maximum time in years
    T = 1

    # Time increments per day
    time_increments_per_day = 100

    # Working days per year
    working_days_per_year = 252

    # Normalisation of the rewards
    # Note: Rewards are additionally scalled by their standard deviation
    reward_normalisation = 5309

    # Costs per treatment (normalised by the same constant as the rewards)
    cost_per_treatment = 4 / reward_normalisation

    # Total budget
    total_budget = 1

    # Empirical cost of treatment
    empirical_cost_of_treatment = 774

    ##
    ## 2. Reward type specific parameters
    ##

    # Input the reward column
    reward_option = int(sys.argv[1])

    # Standard OLS rewards
    if reward_option == 0:

        # Total episodes
        EE = 100000

        # Name of rewards used
        reward_column_name = "Rlr1"

        # Batch size
        batch_size = 1024

        # Evaluation specifics
        evaluation_episodes = 500
        evaluation_every_nn_episodes = 500

    # Doubly robust rewards
    elif reward_option == 1:

        ## Total episodes
        EE = 100000

        # Name of rewards used
        reward_column_name = "d_rob_ols_Xfit"

        # Batch size
        batch_size = 1024

        # Evaluation specifics
        evaluation_episodes = 500
        evaluation_every_nn_episodes = 500

    ##
    ## 3. Parameters chosen as point on grid (learning rates and value basis function)
    ##

    # Grid
    alpha_w_list = [0.001, 0.01, 0.1]
    alpha_theta_list = [0.5, 5, 50]

    # Grid options
    value_int = int(sys.argv[2])
    policy_int = int(sys.argv[3])
    value_basis_int = int(sys.argv[4])

    # Setting the learning rates accordingly
    alpha_w = alpha_w_list[value_int]
    alpha_theta = alpha_theta_list[policy_int]

    # Overwriting the value basis function via class inheritance
    if value_basis_int == 0:

        number_of_states_w = 9

        class ACAgent(ACAgent):
            def return_basis_function_w(self, budget_left, time, T):

                s_w = np.vstack(
                    (
                        (budget_left) * (T - time),
                        (budget_left) * (T - time) ** 2,
                        (budget_left) ** 2 * (T - time),
                        ((budget_left) * (T - time)) ** 2,
                        (budget_left) * np.sin(pi * time),
                        (budget_left) * np.sin(2 * pi * time),
                        (budget_left) ** 2 * np.sin(pi * time),
                        ((budget_left) ** 2 * np.sin(2 * pi * time)),
                        (budget_left) ** 3 * (T - time),
                    )
                )

                return s_w

    elif value_basis_int == 1:

        number_of_states_w = 11

        class ACAgent(ACAgent):
            def return_basis_function_w(self, budget_left, time, T):

                s_w = np.vstack(
                    (
                        (budget_left) * (T - time),
                        (budget_left) * (T - time) ** 2,
                        (budget_left) ** 2 * (T - time),
                        ((budget_left) * (T - time)) ** 2,
                        (budget_left) * np.sin(pi * time),
                        (budget_left) * np.sin(2 * pi * time),
                        (budget_left) ** 2 * np.sin(pi * time),
                        ((budget_left) ** 2 * np.sin(2 * pi * time)),
                        (budget_left) ** 3 * (T - time),
                        (budget_left) ** 3 * np.sin(pi * time),
                        (budget_left) ** 3 * np.sin(2 * pi * time),
                    )
                )

                return s_w

    elif value_basis_int == 2:

        number_of_states_w = 13

        class ACAgent(ACAgent):
            def return_basis_function_w(self, budget_left, time, T):

                s_w = np.vstack(
                    (
                        (budget_left) * (T - time),
                        (budget_left) * (T - time) ** 2,
                        (budget_left) ** 2 * (T - time),
                        ((budget_left) * (T - time)) ** 2,
                        (budget_left) * np.sin(pi * time),
                        (budget_left) * np.sin(2 * pi * time),
                        (budget_left) ** 2 * np.sin(pi * time),
                        ((budget_left) ** 2 * np.sin(2 * pi * time)),
                        (budget_left) ** 3 * (T - time),
                        (budget_left) ** 3 * np.sin(pi * time),
                        (budget_left) ** 3 * np.sin(2 * pi * time),
                        (budget_left) ** 3 * (T - time) ** 2,
                        (budget_left) ** 4 * (T - time),
                    )
                )

                return s_w


    ###################################################################################
    # Parameter settings end
    ###################################################################################


    ##
    ## 4. Data
    ##

    # Create a logger which tracks the process
    programme = os.path.basename(sys.argv[0])
    log_status = logging.getLogger(programme)
    logging.basicConfig(format="%(asctime)s: %(message)s")
    logging.root.setLevel(level=logging.INFO)
    log_status.info("running %s" % " ".join(sys.argv))

    # Set a seed
    np.random.seed(seed=42)

    # Input and output folders

    # Create a unique output folder name with the current date and time
    current_date = datetime.datetime.now()
    current_date = (
        str(current_date)[:-7].replace("-", "_").replace(" ", "_").replace(":", "")
    )

    # Directory (adjust if needed)
    local_directory = "/users/geieckef/dynamic_treatment/"  # on fabian
    #local_directory = "/Users/friedrich/Desktop/"  # laptop

    # Folder paths
    data_directory = os.path.join(local_directory, "data")
    output_directory = os.path.join(
        local_directory, f"outcomes/paper_crosscheck/run_{current_date}_{value_int}_{policy_int}_{value_basis_int}"
    )
    backup_directory = os.path.join(output_directory, "backups")

    # Create folders
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not os.path.exists(backup_directory):
        os.makedirs(backup_directory)

    # Load data

    # RCT data
    df = pd.read_csv(os.path.join(data_directory, "data_lpe2.csv"))

    # Estimated arrival rates
    poisson_table = pd.read_csv(
        os.path.join(data_directory, "sincos_poisson_means_clusters9.csv"), sep="\t"
    )

    # Drop NA rows from the relevant rows of the RCT data
    df = df[["age", "education", "prev_earnings", reward_column_name, "clstrs9"]]
    df = df.dropna()

    # Scale covariates
    scaler = StandardScaler()
    scaler.fit(df.loc[:, ["age", "education", "prev_earnings"]])
    df.loc[:, ["age", "education", "prev_earnings"]] = scaler.transform(
        df.loc[:, ["age", "education", "prev_earnings"]]
    )

    # Subtract the cost from the reward column
    df[reward_column_name] = (
        df[reward_column_name] - empirical_cost_of_treatment
    )  # with 774 being the cost of treatment

    # Preserve the non standardised rewards
    df["rewards_before_standardisation"] = deepcopy(df[reward_column_name])

    # Scale reward by its standard deviation and an additional normalisation
    df[reward_column_name] = df[reward_column_name] / (
        reward_normalisation * np.std(df[reward_column_name])
    )

    # Save all transformed data to disk for later analysis
    df.to_csv(os.path.join(output_directory, f"transformed_rct_data_{reward_column_name}.csv"))

    # The RCT data are split and saved in a dictionary with each cluster being a key
    cluster_dict = {}
    for cc in range(0, 4):
        cluster_dict[cc] = df.loc[
            df.loc[:, "clstrs9"] == (cc + 1),
            ["age", "education", "prev_earnings", reward_column_name],
        ]
        cluster_dict[cc].index = range(0, cluster_dict[cc].shape[0])

    # Next the coefficient estimates from the Poisson regression are loaded
    alpha_vector = poisson_table.loc[
        poisson_table.loc[:, "parm"] == "_cons", "estimate"
    ].values.reshape(4, 1)
    beta_vector = poisson_table.loc[
        poisson_table.loc[:, "parm"] == "dayofyear_sin", "estimate"
    ].values.reshape(4, 1)
    gamma_vector = poisson_table.loc[
        poisson_table.loc[:, "parm"] == "dayofyear_cos", "estimate"
    ].values.reshape(4, 1)

    # Determining the amount of parallel processe to run
    n_parallel_processes_max = max(1, cpu_count())
    n_parallel_processes_selected = 20

    ##
    ## 5. Save all specifications of this run to disk
    ##

    run_specifications = open(
        os.path.join(output_directory, f"run_specifications_{current_date}.txt"), "w"
    )

    agent_for_basis_specs = ACAgent(number_of_states_theta, number_of_states_w)
    lines_basis_function_theta = inspect.getsource(
        agent_for_basis_specs.return_basis_function_theta
    )
    lines_basis_function_w = inspect.getsource(
        agent_for_basis_specs.return_basis_function_w
    )

    run_specifications.write(
        f"""

    Specifications for run {current_date}
    -----------------------------------------


    Parallel processes: {n_parallel_processes_selected} (out of a maximum of {n_parallel_processes_max})


    Rewards used: {reward_column_name} (sys arg {reward_option})


    Learning rate policy function: {alpha_theta}
    Learning rate value function: {alpha_w}


    Batch size: {batch_size}


    Training episodes: {EE}


    Episodes averaged over for each evaluation step: {evaluation_episodes}


    Evaluation every N episodes: {evaluation_every_nn_episodes}


    Number of transformations (+ intercept if defined) used in policy function: {number_of_states_theta}
    Number of transformations (+ intercept if defined) used in value function: {number_of_states_w}



    Basis function with transformations used in the policy function:

    {lines_basis_function_theta}



    Basis function with transformations used in the value function:

    {lines_basis_function_w}



    Total time (in years): {T}


    Total budget: {total_budget}


    Cost per treatment: {cost_per_treatment:.10f}


    Time increments per day: {time_increments_per_day}


    Working days per year: {working_days_per_year}


    Reward normalisation (average arrivals per year): {reward_normalisation}


    Empirical cost of treatment: {empirical_cost_of_treatment}


    Discount rate: {beta:.4f}


    Episodes in which only the value function is updated: {value_update_only_episodes}


    Probability to update the policy function approximator after each batch: {theta_eval_prob}

                 """
    )

    run_specifications.close()

    ##
    ## 6. Running the model
    ##

    # Define multi processing arrays which can be accessed and updated out of all
    # individual processes

    # For the two parameter vectors
    mp_w_array = Array(
        "d", ([0] * number_of_states_w)
    )  # d refers to double precision float
    mp_theta_array = Array("d", ([0] * number_of_states_theta))

    # MP object to track the episode reached in a typical process
    mp_current_episode = Value("d", 0.0)

    # Define a list containing the subprocesses
    processes = [
        Process(
            target=launch_one_actor_critic_process,
            args=(
                mp_w_array,
                mp_theta_array,
                mp_current_episode,
                (process_number + 1),
            ),
        )
        for process_number in range(0, n_parallel_processes_selected)
    ]

    # Start all processes in that list
    for p in processes:

        time_package.sleep(1) # ensures agents are in different parts of the state space
        p.start()

    # Exit any completed processes
    for p in processes:
        p.join()

    # Update that the script is finished
    log_status.info("\n\nTraining completed.\n\n")
