"""
Code for: DYNAMICALLY OPTIMAL TREATMENT ALLOCATION USING REINFORCEMENT LEARNING
by Karun Adusumilli, Friedrich Geiecke, and Claudio Schilter

This code: generate figures VI.1 and VI.5

Part 1: functions
Part 2: simulation (choose setup there accordingly for either figure VI.1 or VI.5)
Part 3: creation of figures


prerequisites: data and policy function
"""
# Importing modules
import matplotlib.pyplot as plt
import seaborn as sns
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
import inspect
import gc
import matplotlib
from sklearn import linear_model
from scipy import sparse as sps

"""
Part 1: functions
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




def get_binscatter_objects(y, x, controls, n_bins, recenter_x, recenter_y, bins):
    """
    Binscatter function from Elizabeth Santorella (taken from github.com/esantorella)
    Returns mean x and mean y within each bin, and coefficients if residualizing.
    Parameters are essentially the same as in binscatter.
    """
    # Check if data is sorted

    if controls is None:
        if np.any(np.diff(x) < 0):
            argsort = np.argsort(x)
            x = x[argsort]
            y = y[argsort]
        x_data = x
        y_data = y
    else:
        # Residualize
        if controls.ndim == 1:
            controls = controls[:, None]

        demeaning_y_reg = linear_model.LinearRegression().fit(controls, y)
        y_data = y - demeaning_y_reg.predict(controls)

        demeaning_x_reg = linear_model.LinearRegression().fit(controls, x)
        x_data = x - demeaning_x_reg.predict(controls)
        argsort = np.argsort(x_data)
        x_data = x_data[argsort]
        y_data = y_data[argsort]

        if recenter_y:
            y_data += np.mean(y)
        if recenter_x:
            x_data += np.mean(x)

    if x_data.ndim == 1:
        x_data = x_data[:, None]
    reg = linear_model.LinearRegression().fit(x_data, y_data)
    if bins is None:
        bin_edges = np.linspace(0, len(y), n_bins + 1).astype(int)
        assert len(bin_edges) == n_bins + 1
        bins = [slice(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
        assert len(bins) == n_bins

    x_means = [np.mean(x_data[bin_]) for bin_ in bins]
    y_means = [np.mean(y_data[bin_]) for bin_ in bins]

    return x_means, y_means, reg.intercept_, reg.coef_[0]


def binscatter(self, x, y, controls=None, n_bins=20,
               line_kwargs=None, scatter_kwargs=None, recenter_x=False,
               recenter_y=True, bins=None):

    if line_kwargs is None:
        line_kwargs = {}
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if controls is not None:
        if isinstance(controls, pd.SparseDataFrame) or isinstance(controls, pd.SparseSeries):
            controls = controls.to_coo()
        elif isinstance(controls, pd.DataFrame) or isinstance(controls, pd.Series):
            controls = controls.values
        assert isinstance(controls, np.ndarray) or sps.issparse(controls)

    x_means, y_means, intercept, coef = get_binscatter_objects(np.asarray(y), np.asarray(x),
                                                               controls, n_bins, recenter_x,
                                                               recenter_y, bins)

    self.scatter(x_means, y_means, **scatter_kwargs)
    x_range = np.array(self.get_xlim())
    self.plot(x_range, intercept + x_range * coef, label='beta=' + str(round(coef, 3)), **line_kwargs)
    # If series were passed, might be able to label
    try:
        self.set_xlabel(x.name)
    except AttributeError:
        pass
    try:
        self.set_ylabel(y.name)
    except AttributeError:
        pass
    return x_means, y_means, intercept, coef

matplotlib.axes.Axes.binscatter = binscatter


"""
Part 2: simulation
"""

if __name__ == "__main__":



    ##
    ## 1. Parameters
    ##

    # Path. This folder has to contain another folder "data" with the datasets "data_lpe2.csv" and "sincos_poisson_means_clusters9.csv"
    local_directory = 'C:/Users/ClaudioSchilter/Dropbox/Reinforcement_learning/'
    
    ## for DR rewards
    folder = 'fabian_outcomes_to_analyse/paper/run_2020_07_21_001221_1_1_0/'
    
    ## for std OLS rewards
#    folder = 'fabian_outcomes_to_analyse/paper/run_2020_07_21_001159_1_1_0/'

    # Evaluation episodes
    EE = 1000

    sampled_actions = True
    policy_description = "110_DR"
    dataset = pd.read_csv(local_directory+folder+"policy_parameter_path.csv",header=None)
    policy_parameters = dataset[-1:].values.tolist()[0][1:13]


    # Rewards considered
    
    ## for std OLS rewards
#    reward_column_name = "Rlr1"
    
    ## for DR rewards
    reward_column_name = "d_rob_ols_Xfit"

    # Number of states
    number_of_states_theta = 12
    number_of_states_w = 9

    # Discount factor
    beta =  - np.log(0.9)
#    beta =  0 # can set discounting to zero for KT comparison

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
    data_directory = os.path.join(local_directory, 'evaluation/data')


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

    # Then the RCT data are split and saved in a dictionary with each cluster being a key
    cluster_dict = {}
    for cc in range(0, 4):
        #cluster_dict[cc] = df.loc[df.loc[:,'clstrs9'] == (cc+1), ['age', 'education', 'prev_earnings', reward_column_name, 'compProp']]
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


    # Setting the parameters in the agent instance
    function_approximators_object.theta[:] = policy_parameters

    # Evaluation rewards achieved
    cumulative_episode_rewards = []

    
    output_file = open(local_directory+folder+"simulation_outcomes.csv","w")
    output_file.write('time, budget, skipped, iteration \n')
    
    # Begin the loop over episodes
    ee = 0
    while ee < EE:
        
        print(ee+1)
        
        # Fixing the seed
        np.random.seed(ee+1)

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
        skip = 0

        # Initialise with first person from first cluster
        s0 = cluster_dict[0].loc[0,['age', 'education', 'prev_earnings']].values.reshape(3,1)
        s0_prime = deepcopy(s0)
        
        # Building basis function
        s_theta = function_approximators_object.return_basis_function_theta(s0 = s0, budget_left = budget_left, time = time, T = T)

        # Treatment reward
        s_R_if_treated = cluster_dict[0].loc[0,reward_column_name]

        # Re-setting the terminal flag to False
        terminal_flag = False
        
        counter=0
        
        if ee % 10 ==0:
            output_file.flush()
            gc.collect()

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
                skip += 1

            # Otherwise update reward, people treated, and budget left
            elif a == 1:

                R = s_R_if_treated
                
                line_item = str(time)+","+str(budget_left)+","+str(skip)+","+str(ee+1)
                output_file.write(line_item + "\n")
                
                counter += 1
                people_treated += 1
                budget_left -= cost_per_treatment                
                skip = 0

            # Update episode reward
            cumulative_episode_reward += I*R

            # Updating the people count within the episode
            people_considered_for_treatment += 1

            # For each cluster, draw the expected number of people arriving per 100th of a day
            lambda_vector = (np.exp(alpha_vector + (beta_vector * np.sin(time)) + (gamma_vector * np.cos(time))) / lambda_normalisation).reshape(4,1)

            # Summed lambda vector: The expected number of people (irrespective of clusters) arriving in a 100th of a day
            summed_lambda_vector = np.sum(lambda_vector)

            # Updating time (t' and budget_left' are already needed for the function approximator updates of this period)

            delta_time = (math.ceil(np.random.exponential(scale=1/summed_lambda_vector, size=None)) / time_grid_normalisation)
            time = time + delta_time

            # Creating discounting factor gamma that is time increment specific (discounting is actually only updated after the function approximator updates of that episode)
            gamma = np.exp(- beta * delta_time)

            # Drawing the cluster from which the next person arrives
            cluster = np.dot(np.array([0, 1, 2, 3]), np.random.multinomial(1, pvals = (lambda_vector/summed_lambda_vector).reshape(4,)))

            # Drawing the next individual
            sample_person_index = int(np.random.choice(range(0,cluster_dict[cluster].shape[0]), size=1))
            s0_prime = cluster_dict[cluster].loc[sample_person_index,['age', 'education', 'prev_earnings']].values.reshape(3,1)

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

    output_file.close()


"""
Part 3: figures
if simulation above already run, can also start simply from here
"""
   
    saverframe = pd.read_csv(local_directory+folder+"simulation_outcomes.csv")
    
    saverframe['Month'] = "December"
    saverframe.loc[saverframe['time']<11/12,'Month'] = "November"
    saverframe.loc[saverframe['time']<10/12,'Month'] = "October"
    saverframe.loc[saverframe['time']<9/12,'Month'] = "September"
    saverframe.loc[saverframe['time']<8/12,'Month'] = "August"
    saverframe.loc[saverframe['time']<7/12,'Month'] = "July"
    saverframe.loc[saverframe['time']<6/12,'Month'] = "June"
    saverframe.loc[saverframe['time']<5/12,'Month'] = "May"
    saverframe.loc[saverframe['time']<4/12,'Month'] = "April"
    saverframe.loc[saverframe['time']<3/12,'Month'] = "March"
    saverframe.loc[saverframe['time']<2/12,'Month'] = "February"
    saverframe.loc[saverframe['time']<1/12,'Month'] = "January"
    
    stdframe = saverframe[[' skipped', 'Month']].groupby(['Month']).std()
    countframe = saverframe[[' skipped', 'Month']].groupby(['Month']).count()
    monthframe = saverframe[[' skipped', 'Month']].groupby(['Month']).mean()
    monthframe['Month'] = monthframe.index
    monthframe['Count'] = countframe[' skipped']
    monthframe['SD'] = stdframe[' skipped']
    monthframe['height_errorbars'] = 1.96*monthframe['SD']/np.sqrt(monthframe['Count'])
    monthframe = monthframe.reindex(["January","February","March","April","May","June","July","August","September","October","November","December"])
    
    if math.isnan(monthframe['Count'][11]):
        monthframe=monthframe.drop('December')
        
    
    plt.style.use('seaborn')
#    plt.rc('font', family='serif')
    plt.rcParams["font.family"] = "serif"
#   plt.rcParams["mathtext.fontset"] = "dejavuserif"

    palette_used = sns.color_palette("husl", 12)
    barWidth = 0.8
    # Choose the height of the bars
    bars1 = list(monthframe[' skipped'])
     
    
    # Choose the height of the error bars
    yer1 = list(monthframe['height_errorbars'])
     
    # The x position of bars
    r1 = np.arange(len(bars1))
    
    r1 = list(monthframe['Month'])
    
    plt.rcParams.update({'font.size': 10})
    


    
    # Create bars
    fig = plt.bar(r1, bars1, width = barWidth, color = palette_used[8], yerr=yer1, capsize=7)
#    plt.figure(2, figsize=(17,17))
    
    matplotlib.rcParams.update({'font.size': 10})
    plt.ylabel('Avg # Untreated Applicants before Treatment',fontsize=10,family='serif')
    plt.xticks(rotation=45,fontsize=10,family='serif')
    plt.yticks(fontsize=10)
    
    plt.savefig(local_directory+folder+"pickiness_bymonth.pdf")
      
    
    
    fig, bscatter = plt.subplots(1)
    bscatter.binscatter(saverframe[' budget'], saverframe[' skipped'])
    bscatter.set_xlabel("Remaining Budget")
    bscatter.set_ylabel("Avg # Untreated Applicants before Treatment")
    plt.xlim(1,0)
    plt.savefig(local_directory+folder+"bscatter.pdf")
    


