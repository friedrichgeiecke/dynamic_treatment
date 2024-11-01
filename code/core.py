# Code from: Spinning Up in Deep Reinforcement Learning (https://github.com/openai/spinningup), Joshua Achiam (2018)
#
# Adjusted to solve RCT environments in Adusumilli, Geiecke, Schilter (2024)

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


# Logistic/sigmoid policy
class LogisticCategoricalActor(Actor):
    def __init__(self, num_policy_states, non_stationary_policy):
        super().__init__()
        self.logits_net = nn.Sequential(nn.Linear(num_policy_states, 1), nn.Identity())
        self.num_policy_states = num_policy_states
        self.non_stationary_policy = non_stationary_policy
        if self.non_stationary_policy:

            self.num_covariates = (
                self.num_policy_states - 1 - 11
            )  # minus budget, minus 11 monthly dummies

            # Indices of states used in non-stationary policy
            # For two covariates, this would e.g. yield [0, 1, 2, 4, ..., 14]
            self.non_stationary_covariate_indices = (
                list(range(self.num_covariates))
                + [self.num_covariates]  # budget
                + list(
                    range(self.num_covariates + 2, self.num_covariates + 13)
                )  # 11 monthly dummies b/c of intercept in model
            )

    def _distribution(self, obs):

        # For nonstationary policy, also use all policy states of environment (including
        # budget and time)
        if self.non_stationary_policy:

            if obs.dim() == 2:
                logits = self.logits_net(obs[:, self.non_stationary_covariate_indices])
            else:
                logits = self.logits_net(obs[self.non_stationary_covariate_indices])

        # For stationary policy, only use individual covariates in policy (stored as
        # first three states)
        else:

            if obs.dim() == 2:
                logits = self.logits_net(obs[:, : self.num_policy_states])
            else:
                logits = self.logits_net(obs[: self.num_policy_states])

        logits = torch.squeeze(logits, -1)
        return Bernoulli(logits=logits)  # implicitly applies sigmoid

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPCategoricalActor(Actor):
    def __init__(
        self,
        num_policy_states,
        num_actions,
        hidden_sizes,
        activation,
        non_stationary_policy,
    ):
        super().__init__()
        self.logits_net = mlp(
            [num_policy_states] + list(hidden_sizes) + [num_actions], activation
        )
        self.num_policy_states = num_policy_states
        self.non_stationary_policy = non_stationary_policy

    def _distribution(self, obs):

        # For nonstationary policy, also use all policy states of environment (including
        # budget and time)
        if self.non_stationary_policy:

            logits = self.logits_net(obs)

        # For stationary policy, only use individual covariates in policy (stored as
        # first three states)
        else:

            if obs.dim() == 2:
                logits = self.logits_net(obs[:, : self.num_policy_states])
            else:
                logits = self.logits_net(obs[: self.num_policy_states])

        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPCritic(nn.Module):
    def __init__(self, num_value_states, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([num_value_states] + list(hidden_sizes) + [1], activation)
        self.num_value_states = num_value_states

    def forward(self, obs):

        if obs.dim() == 2:
            value = self.v_net(
                obs[:, : self.num_value_states]
            )  # policy might have more states if logistic and time dummies
        else:
            value = self.v_net(obs[: self.num_value_states])

        return torch.squeeze(value, -1)  # ensures v has right shape


class ActorCritic(nn.Module):
    def __init__(
        self,
        num_policy_states,
        num_value_states,
        num_actions,
        non_stationary_policy,
        logistic_policy,
        hidden_sizes=(
            64,
            64,
        ),  # affect only value function if policy is set to logistic
        activation=nn.Tanh,
    ):
        super().__init__()

        # Store policy type (used in step function)
        self.logistic_policy = logistic_policy

        # Policy function
        if self.logistic_policy:
            self.pi = LogisticCategoricalActor(
                num_policy_states=num_policy_states,
                non_stationary_policy=non_stationary_policy,
            )
        else:

            self.pi = MLPCategoricalActor(
                num_policy_states=num_policy_states,
                num_actions=num_actions,
                hidden_sizes=hidden_sizes,
                activation=activation,
                non_stationary_policy=non_stationary_policy,
            )

        # Value function
        self.v = MLPCritic(
            num_value_states=num_value_states,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def step(self, obs, deterministic=False):
        # Evaluate the policy and value functions for a given state
        with torch.no_grad():

            pi = self.pi._distribution(obs)

            if deterministic:

                if self.logistic_policy:

                    a = (pi.probs > 0.5).float()

                else:

                    a = pi.probs.argmax()

            else:
                a = pi.sample()

            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs, deterministic=False):
        return self.step(obs, deterministic=deterministic)[0]
