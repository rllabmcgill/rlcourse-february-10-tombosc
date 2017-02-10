import numpy as np
import matplotlib.pyplot as plt

def step(s, a, p_h, n):
    """
    simulate a step

    agent is in state s, bet a where 1 <= a <= s
    returns (next state, reward)
    """
    assert(a>= 1 and a<=s)
    rand = np.random.choice(2, p=[1-p_h, p_h])
    if rand == 1:
        next_s = s + a
    else:
        next_s = s - a

    reward = 0
    if next_s >= n:
        reward = 1
        next_s = n
    elif next_s <= 0:
        next_s = 0
    return (next_s, reward)

def greedify(pi):
    """
    returns greedified policy pi where pi[a,s] is a policy pi(a|s)
    """
    amax = pi.argmax(axis = 0)
    greedy = np.zeros(pi.shape)
    greedy[amax,range(pi.shape[1])] = 1
    return greedy

def epsilon_greedify(pi, epsilon):
    """
    returns epsilon-greedified policy pi where pi[a,s] is a policy pi(a|s)

    no need to first greedify it
    """
    pi = greedify(pi)
    actions_max = np.argmax(pi[:,2:], axis=0)
    mu = np.copy(pi)
    for s, amax in zip(range(2,pi.shape[1]),actions_max):
        mu[1:s+1, s] += epsilon / s
        mu[amax, s] -= epsilon * (1 + (1/(s)))
    assert(np.allclose(mu.sum(axis=0), np.ones(pi.shape[0])))
    return mu
 
