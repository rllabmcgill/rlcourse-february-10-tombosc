import numpy as np
import matplotlib.pyplot as plt
from utils import step, greedify, epsilon_greedify

"""
Sarsa and Expected Sarsa on the gambler's problem.

the gambler has to gamble at least 1, the discount rate is 1.
"""

n = 2**5 # goal: when attained, end of episode
p_h = 0.3 # probability of winning double the bet
# The task is harder with a low p_h as it will get a lot less reward quickly

def sarsa(max_iter, alpha, eps, expected=False, start_from=None):
    """
    TD Control using Sarsa
    max_iter: number of updates (not episodes!)
    alpha: learning rate
    eps: epsilon used during epsilon-greedification of policy
    expected: if true, use the Expected Sarsa variant
    start_from: - None: exploring starts
                - (state, None): start from state, pick random action
                - (state, action): 
    """
    # init random policy pi[a,s]
    pi = np.zeros((n+1,n+1))
    for s in range(0, n+1):
        for a in range(1, s+1):
            pi[a,s] = np.random.uniform()
        pi[:,s] = pi[:,s] / pi[:,s].sum()
    pi = np.nan_to_num(pi)
    
    q = np.zeros((n+1,n+1)) 

    sample_action = lambda s, pi: np.random.choice(n+1, p=pi[:,s]) # from 1 to s
    random_initial_state = lambda: np.random.choice(n-1) + 1 # from 1 to n-1

    def init_s_a():
        """
        helper to initialize randomly s and a based on start_from 
        """
        if start_from:
            s, a = start_from
            if a == None:
                a = sample_action(s, pi)
        else:
            s = random_initial_state()
            a = sample_action(s, pi)
        return (s,a)

    s,a = init_s_a()

    pi = epsilon_greedify(pi, eps) 
    n_episode = 1
    for i in range(1,max_iter):
        s_n, r = step(s, a, p_h, n)
        # it's OK to call epsilon_greedify on the q array
        pi[1:,1:] = epsilon_greedify(q[1:,1:], eps) 
        a_n = sample_action(s_n, pi)
        if expected:
            q[a,s] += alpha * (r + np.dot(q[:,s_n], pi[:, s_n]) - q[a,s])
        else:
            q[a,s] += alpha * (r + q[a_n,s_n] - q[a,s])

        if s_n == 0 or s_n == n: # start new episode
            s,a = init_s_a()
            n_episode += 1
        else:
            s = s_n
            a = a_n

    #print "total episode seen:", n_episode
    return q, greedify(q)

for alpha in np.arange(0.1, 1.0, 0.1): # grid search for learning rate
    max_iter = 300000
    eps = 0.05
    q, pi = sarsa(max_iter, alpha, eps, expected=False, start_from=None)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14,5))
    im_1 = ax1.imshow(q[1:,1:n], interpolation='None', origin='lower',
              extent=(1,n-1,1,n))
    f.colorbar(im_1, ax=ax1)

    ax1.set_title("q(a,s)")
    im_2 = ax2.imshow(pi[1:,1:n], interpolation='None', origin='lower',
              extent=(1,n-1,1,n))
    f.colorbar(im_2, ax=ax2)
    ax2.set_title("Target policy pi")
    #plt.show()
    f.savefig("gambler_sarsa_"+str(max_iter)+"_it_p_03_alph_" + 
              str(alpha) + "_eps_"+str(eps)+".png")
