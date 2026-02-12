import numpy as np

# global stopping criteria
EPS = 0.001

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




    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    return val_, pi

def policy_iteration(model, maxiter):
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
    pi = np.ones((model.num_states, 1))
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








        #######################################################################
        #                       END OF YOUR CODE                              #
        #######################################################################

        # check if stopping criteria satisfied
        if stable_policy:
            print("Policy iteration converged after %d iterations." % i)
            break

    return val_, pi

def policy_evaluation(model, val_, policy):
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









        #######################################################################
        #                       END OF YOUR CODE                              #
        #######################################################################

        # stopping criteria
        if delta <= EPS * (1 - model.gamma) / model.gamma:
            loop = False

    return val_
