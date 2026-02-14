import numpy as np
from rl_env.grid_world import GridWorld

# global stopping criteria
EPS = 0.001

def compute_qvalues(R: np.ndarray, gamma: float, P: np.ndarray, values: np.ndarray):
    """
    R is (n_states, 1). P is (n_states, n_states, n_actions). values is (n_states, 1).
    Returns q-values: (n_states, n_actions).
    """
    # (n_states, n_states, n_actions) x (n_states,) -> (n_states, n_actions).
    return R + gamma * np.tensordot(P, values[:, 0], axes=(1, 0))

def value_iteration(model, maxiter=100):
    """
    Solves the supplied environment with value iteration.

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.

    maxiter : int
        The maximum number of iterations to perform.

    Return
    ------
    val_ : numpy array of shape (N, 1)
        Value function of the environment where N is the number
        of states in the environment.

    pi : numpy array of shape (N, 1)
        Optimal policy of the environment.
    """
    # initialize the value function and policy
    pi = np.ones((model.num_states, 1))
    val_ = np.zeros((model.num_states, 1))

    for i in range(maxiter):
        # initialize delta
        delta = 0
        # perform Bellman update for each state

        #######################################################################
        # TODO1:                                                              #
        # Perform Bellman optimality update for each state                    #
        # delta stores the maximum value change after update over all states, #
        # which is used to determine the convergence of VI.                   #
        #######################################################################
        q = compute_qvalues(model.R, model.gamma, model.P, val_)
        new_values = q.max(axis=-1, keepdims=True)
        delta = np.abs(new_values - val_).max()
        val_ = new_values
        #######################################################################
        #                       END OF YOUR CODE                              #
        #######################################################################

        # stopping criteria
        if delta <= EPS * (1 - model.gamma) / model.gamma:
            print("Value iteration converged after %d iterations." %  i)
            break


    #####################################################################
    # TODO2:                                                            #
    # Compute the optimal policy from value function                    #
    # i.e., update pi from val_                                         #
    #####################################################################
    pi = q.argmax(axis=-1)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    return val_, pi

def policy_iteration(model: GridWorld, maxiter):
    """
    Solves the supplied environment with policy iteration.

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.

    maxiter : int
        The maximum number of iterations to perform.

    Return
    ------
    val_ : numpy array of shape (N, 1)
        Value function of the environment where N is the number
        of states in the environment.

    pi : numpy array of shape (N, 1)
        Optimal policy of the environment.
    """
    # initialize the value function and policy
    pi = np.ones((model.num_states, 1), dtype=np.int64)
    val_ = np.zeros((model.num_states, 1))

    for i in range(maxiter):
        # Stopping criteria
        stable_policy = True
        # Policy evaluation
        val_ = policy_evaluation(model, val_, pi)

        #######################################################################
        # TODO3:                                                              #
        # Perform greedy policy improvement for each state.                   #
        # Set "stable_policy" to False, if there is any policy update over    #
        # all states, which is used to determine the convergence of PI.       #
        #######################################################################
        q = compute_qvalues(model.R, model.gamma, model.P, val_)
        new_policy = q.argmax(axis=-1, keepdims=True)
        stable_policy = (pi == new_policy).all()
        pi = new_policy
        #######################################################################
        #                       END OF YOUR CODE                              #
        #######################################################################

        # check if stopping criteria satisfied
        if stable_policy:
            print("Policy iteration converged after %d iterations." % i)
            break

    return val_, pi

def policy_evaluation(model: GridWorld, val_, policy):
    """
    Evaluates a given policy.

    Parameters
    ----------
    model : python object
       Holds information about the environment to solve
       such as the reward structure and the transition dynamics.

    val_ : numpy array of shape (N, 1)
       Value function of the environment where N is the number
       of states in the environment.

    policy : numpy array of shape (N, 1)
       Optimal policy of the environment.

    Return
    ------
    val_ : numpy array of shape (N, 1)
       Value function of the environment where N is the number
       of states in the environment.
    """
    loop = True
    while loop:
        # initialize delta
        delta = 0

        #######################################################################
        # TODO4:                                                              #
        # Perform Bellman expectation  update for each state                  #
        # delta stores the maximum value change after update over all states, #
        # which is used to determine the convergence of policy evaluation.    #
        #######################################################################
        q = compute_qvalues(model.R, model.gamma, model.P, val_)
        new_vals = q[np.arange(q.shape[0]), policy[:, 0]][:, None]
        delta = np.abs(new_vals - val_).max()
        val_ = new_vals
        #######################################################################
        #                       END OF YOUR CODE                              #
        #######################################################################

        # stopping criteria
        if delta <= EPS * (1 - model.gamma) / model.gamma:
            loop = False

    return val_
