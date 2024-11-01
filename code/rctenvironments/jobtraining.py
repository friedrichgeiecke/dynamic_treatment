# Dynamic job training environment from Adusumilli, Geiecke, Schilter (2024)

import numpy as np
import os
import pandas as pd
import rctenvironments.ite_estimation as itee
from sklearn.preprocessing import StandardScaler


# Helper function to load data
def load_and_merge_data(input_dir):
    """

    Load and merge data from Kitagawa and Tetenov (2018) & the JTPA National Evaluation.

    Inputs

    input_dir: path of input directory (string)

    Output

    Merged data frame

    """

    # Import data from Kitagawa and Tetenov (2018) (supplemental materials from the
    # Econometric Society)
    jtpa_kt = pd.read_csv(os.path.join(input_dir, "jtpa_kt.tab"), delimiter="\t")
    jtpa_kt = jtpa_kt.rename(
        columns={
            "D": "allocated_trmt",
            "edu": "education",
            "prevearn": "prev_earnings",
        }
    )
    # Import date and age variables from the JTPA National Evaluation (Upjohn Institute)
    expbif = pd.read_stata(os.path.join(input_dir, "expbif.dta"))
    expbif = expbif.rename(columns={"ra_dt": "date"})
    expbif = expbif[["recid", "date", "age"]].astype(
        {"age": "float64", "recid": "int64"}
    )
    # Convert date to datetime
    expbif["date"] = pd.to_datetime(expbif["date"], format="%y%m%d")

    # Merge the two datasets
    df = pd.merge(expbif, jtpa_kt, on="recid", how="inner")

    # Return merged data frame
    return df


class DynamicJTPA:
    def __init__(
        self,
        input_dir,
        output_dir,
        non_stationary_policy,
        logistic_policy,
        kt_policy_normalisation=False,
        normalise_z_t=True,
        T=1.0,
        budget_sufficient_to_treat_fraction_of_yearly_arrivals=0.25,
        discount_factor=1.0,
        empirical_cost_of_treatment=774.0,
        earnings_lower_threshold=1257.0,
        earnings_upper_threshold=5001.0,
        kt_rewards=False,
        individual_specific_costs=True,
        monthly_arrival_rates=True,
        start_month=0,
        covariates=[
            "age",
            "education",
            "prev_earnings",
        ],
        outcome="earnings",
        treatment_indicator="allocated_trmt",
        write_processed_env_data_to_disk=False,
        seed_overall=None,
    ):

        # Set if not none
        if seed_overall is not None:
            np.random.seed(seed_overall)

        # Store input parameters as attributes
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.non_stationary_policy = non_stationary_policy
        self.logistic_policy = logistic_policy
        self.kt_policy_normalisation = kt_policy_normalisation
        self.normalise_z_t = normalise_z_t
        self.T = T
        self.Z = 1  # normalised to 1
        self.discount_factor = discount_factor
        self.budget_sufficient_to_treat_fraction_of_yearly_arrivals = (
            budget_sufficient_to_treat_fraction_of_yearly_arrivals
        )
        self.empirical_cost_of_treatment = empirical_cost_of_treatment
        self.earnings_lower_threshold = earnings_lower_threshold
        self.earnings_upper_threshold = earnings_upper_threshold
        self.kt_rewards = kt_rewards
        self.individual_specific_costs = individual_specific_costs
        self.monthly_arrival_rates = monthly_arrival_rates
        self.start_month = (
            start_month  # 0/12: Beginning of January, 1/12: Beginning of February, etc.
        )
        self.start_month_normalised = self.start_month / 12
        self.months_array = np.arange(11)  # stored once to compute monthly dummies
        # When using KT 2018 rewards, always only use education and prev_earnings
        if self.kt_rewards:
            self.covariates = ["education", "prev_earnings"]
        else:
            self.covariates = covariates
        self.num_covariates = len(self.covariates)
        self.outcome = outcome
        self.treatment_indicator = treatment_indicator
        self.write_processed_env_data_to_disk = write_processed_env_data_to_disk
        self.seed_overall = seed_overall

        # Dimensions

        # Covars + budget + time
        self.num_buffer_states = self.num_covariates + 2

        # Nonstationary policy
        if self.non_stationary_policy:

            # Add monthly dummies for logistic policy
            if self.logistic_policy:

                self.num_policy_states = (
                    self.num_covariates + 1 + 11
                )  # covars + z + 11 monthly dummies

                # Buffer also stores 11 dummies in that case
                self.num_buffer_states += 11

            # For neural nets, just keep plain 5 states
            else:

                self.num_policy_states = (
                    self.num_covariates + 2
                )  # covars + budget + time
        # Stationary policy
        else:
            self.num_policy_states = self.num_covariates  # only covars
        # Value function is always a neural network and includes budget and time
        self.num_value_states = self.num_covariates + 2
        # Number of actions
        self.num_actions = 2

        # Load and merge JTPA data from KT 2018 replication package and Upjohn Institute
        df = load_and_merge_data(self.input_dir)

        # Delete NAs if any
        df = df.dropna(
            subset=self.covariates + [self.outcome, self.treatment_indicator]
        )

        ## Rewards

        # Use KT 2018 ITEs
        if self.kt_rewards:

            print("Using ITE estimates from KT 2018")

            # Read original KT data
            kt_ites = pd.read_csv(os.path.join(self.input_dir, "ites_kt.csv"))
            kt_ites = kt_ites.rename(columns={"ite": "reward"})

            # Merge into main data
            df = pd.merge(df, kt_ites[["recid", "reward"]], on="recid", how="inner")

        # Estimate own doubly robust ITEs
        else:

            print("Estimating doubly robust ITEs")

            # Estimate ITEs
            df = itee.add_doubly_robust_ites(
                df=df,
                outcome=self.outcome,
                treatment_indicator=self.treatment_indicator,
                covariates=self.covariates,
            )

            # Subtract empirical cost of treatment
            df["reward"] = df["ite"] - self.empirical_cost_of_treatment

        # Scale rewards
        scaler_r = StandardScaler(with_mean=False, with_std=True)
        scaler_r.fit(df[["reward"]])
        df.loc[:, ["reward_scaled"]] = scaler_r.transform(df[["reward"]])

        ## Arrival rates

        # Extract year and month from the date
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        # Average monthly arrivals
        self.monthly_arrivals = (
            df.groupby(["year", "month"])
            .size()
            .reset_index(name="count")
            .groupby("month")["count"]
            .mean()
            .tolist()
        )
        # Average monthly arrivals scaled to yearly values
        self.yearly_arrivals_implied_by_month = [a * 12 for a in self.monthly_arrivals]
        # Combined arrivals in a year
        self.combined_yearly_arrivals = sum(self.monthly_arrivals)
        # Time increment for constant arrival rate case (year normalised to 1)
        self.constant_delta_time = 1 / self.combined_yearly_arrivals

        ## Costs

        # Individual-specific treatment costs
        if self.individual_specific_costs:

            df["cost"] = (
                (df["prev_earnings"] / 5 <= self.earnings_lower_threshold)
                * self.earnings_lower_threshold
                + (
                    (df["prev_earnings"] / 5 > self.earnings_lower_threshold)
                    & (df["prev_earnings"] / 5 < self.earnings_upper_threshold)
                )
                * df["prev_earnings"]
                / 5
                + (df["prev_earnings"] / 5 >= self.earnings_upper_threshold)
                * self.earnings_upper_threshold
            )

            # Average costs by month
            self.average_individual_costs_by_month = (
                df.groupby("month")["cost"].mean().tolist()
            )

            # Average combined costs of all arrivals in a year
            self.combined_costs_of_yearly_arrivals = sum(
                [
                    self.monthly_arrivals[i] * self.average_individual_costs_by_month[i]
                    for i in range(12)
                ]
            )

            # Scale costs such that a budget of 1 is able to treat
            # ~self.budget_sufficient_to_treat_fraction_of_yearly_arrivals
            df.loc[:, ["cost_scaled"]] = (
                df["cost"]
                / self.budget_sufficient_to_treat_fraction_of_yearly_arrivals
                / self.combined_costs_of_yearly_arrivals
            )
        else:
            # Constant costs scaled such that a budget of 1 is able to treat
            # ~self.budget_sufficient_to_treat_fraction_of_yearly_arrivals
            df["cost_scaled"] = (
                1
                / self.budget_sufficient_to_treat_fraction_of_yearly_arrivals
                / self.combined_yearly_arrivals
            )

        # Standardise/scale covariates
        self.covariates_scaled = [f"{cov}_scaled" for cov in self.covariates]
        if kt_policy_normalisation:

            df[self.covariates_scaled] = df[self.covariates] / df[self.covariates].max()

        else:

            scaler = StandardScaler()
            df[self.covariates_scaled] = scaler.fit_transform(df[self.covariates])

        # Shifting t&z such that they have closer to zero mean, similarly to covariates
        if self.normalise_z_t:
            self.z_state_shifter = -(
                self.Z / 2
            )  # if initial budge is normalised to Z=1, this makes z run from 0.5 to -0.5
            self.t_state_shifter = -(
                self.T / 2
            )  # if initial time is 0 and maximum time T=1, this makes t run from -0.5 to 0.5
        else:
            self.z_state_shifter = 0
            self.t_state_shifter = 0

        ## Store data of simulator as attribute

        # Full processed environment data
        self.df = df

        # Then store small version only with environment states, rewards, and costs

        # If automatic saving to disk is enabled, save data
        if self.write_processed_env_data_to_disk:
            self.save_data()

        if self.monthly_arrival_rates:

            self.data_by_month = {}
            for m in range(12):
                self.data_by_month[m] = df.loc[
                    df.loc[:, "date"].dt.month == (m + 1),
                    self.covariates_scaled + ["reward_scaled", "cost_scaled"],
                ].reset_index(drop=True)

        else:

            self.data_full = df[
                self.covariates_scaled + ["reward_scaled", "cost_scaled"]
            ]

    def save_data(self, file_extension=""):

        self.df.to_csv(
            os.path.join(self.output_dir, f"transformed_rct_data{file_extension}.csv")
        )

    def reset(self, seed_episode=None):

        # Condition added since otherwise np.random.seed(None) overwrites previous seed
        if seed_episode is not None:
            np.random.seed(seed_episode)

        # Reset budget, time, done flag, and episode data
        # Time
        self.t = 0
        # Budget
        self.z = self.Z
        # Done flag
        self.done = False
        # Rewards if treated
        self.potentialrs = []
        # Treatment costs if treated
        self.potentialcs = []
        # Actual rewards (either potentialr if treated or 0)
        self.rs = []
        # Actual costs (either potentialc if treated or 0)
        self.cs = []
        # Arrival times
        self.ts = []

        # Draw time increment of first arrival
        if self.monthly_arrival_rates:

            delta_time = np.random.exponential(
                scale=1 / self.yearly_arrivals_implied_by_month[0], size=None
            )

        else:

            # Constant arrival time increment
            delta_time = self.constant_delta_time

        # Update time and store it
        self.t += delta_time
        self.ts.append(self.t)

        # Draw first observation
        if self.monthly_arrival_rates:

            # Draw first observation
            obs0 = self.data_by_month[int(self.start_month_normalised * 12)].sample(n=1)

        else:

            # Draw first observation from full data
            obs0 = self.data_full.sample(n=1)

        potentialr0 = self.discount_factor**self.t * obs0["reward_scaled"].iloc[0]
        potentialc0 = obs0["cost_scaled"].iloc[0]
        self.potentialrs.append(potentialr0)
        self.potentialcs.append(potentialc0)

        # Construct state

        # For nonstationary logistic policy
        if self.non_stationary_policy and self.logistic_policy:

            s0 = np.hstack(
                [
                    obs0[self.covariates_scaled].values.reshape(
                        self.num_covariates,
                    ),
                    (self.z + self.z_state_shifter),
                    (self.t + self.t_state_shifter),
                    (self.months_array == 0).astype(int),
                ]
            )

        # For all other policies (state always includes z & t for value function)
        else:

            s0 = np.hstack(
                [
                    obs0[self.covariates_scaled].values.reshape(
                        self.num_covariates,
                    ),
                    (self.z + self.z_state_shifter),
                    (self.t + self.t_state_shifter),
                ]
            )

        # Return state
        return s0

    def step(self, action):
        # 1. Determine reward and adjust budget
        if action == 0:
            r = 0
            c = 0

        elif action == 1:
            r = self.potentialrs[-1]
            c = self.potentialcs[-1]
        else:
            raise ValueError("Invalid action")

        # Update budget
        self.z -= c

        # Store reward and cost
        self.rs.append(r)
        self.cs.append(c)

        # 2. Adjust time and and determine next state

        # Determine month
        month = int((self.start_month_normalised + self.t) * 12) % 12

        # Draw the time increment until the next arrival
        if self.monthly_arrival_rates:

            delta_time = np.random.exponential(
                scale=1 / self.yearly_arrivals_implied_by_month[month],
                size=None,
            )

        else:

            delta_time = self.constant_delta_time

        # Update time and store it
        self.t += delta_time
        self.ts.append(self.t)

        if self.monthly_arrival_rates:

            # Draw first observation
            obsprime = self.data_by_month[month].sample(n=1)

        else:

            # Draw first observation from full data
            obsprime = self.data_full.sample(n=1)

        potentialrprime = (
            self.discount_factor**self.t * obsprime["reward_scaled"].iloc[0]
        )
        potentialcprime = obsprime["cost_scaled"].iloc[0]
        self.potentialrs.append(potentialrprime)
        self.potentialcs.append(potentialcprime)

        # Construct state

        # For nonstationary logistic policy
        if self.non_stationary_policy and self.logistic_policy:

            sprime = np.hstack(
                [
                    obsprime[self.covariates_scaled].values.reshape(
                        self.num_covariates,
                    ),
                    (self.z + self.z_state_shifter),
                    (self.t + self.t_state_shifter),
                    (self.months_array == month).astype(int),
                ]
            )

        # For all other policies (state always includes z & t for value function)
        else:

            sprime = np.hstack(
                [
                    obsprime[self.covariates_scaled].values.reshape(
                        self.num_covariates,
                    ),
                    (self.z + self.z_state_shifter),
                    (self.t + self.t_state_shifter),
                ]
            )

        # 3. Determine whether episode terminates because next arrival's time or budget
        # is beyond constraint
        if self.t >= self.T or self.z <= 0:
            self.done = True

        # Return next state, current reward achieved through action, and whether episode
        # ends
        return sprime, r, self.done
