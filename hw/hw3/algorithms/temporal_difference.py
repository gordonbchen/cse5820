import numpy as np

def sarsa(model, alpha=0.5, min_epsilon=0.1, maxiter=100, maxeps=1000):
    """
    Solves the supplied environment using SARSA.

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.

    alpha : float
        Algorithm learning rate. Defaults to 0.5.

    min_epsilon : float
         epsilon is the probability that a random action is selected.
         epsilon must be in the interval [0,1] where 0 means that the
         action is selected in a completely greedy manner and 1 means
         the action is always selected randomly. Here min_epsilon is
         the minimal epsilon for epsilon decay. In practice, epsilon
         is in the interval [min_epsilon, 1].

    maxiter : int
        The maximum number of iterations to perform per episode.
        Defaults to 100.

    maxeps : int
        The number of episodes to run SARSA for.
        Defaults to 1000.

    Returns
    -------
    pi : numpy array of shape (N, 1)
        Optimal policy for the environment where N is the total
        number of states.

    state_counts : numpy array of shape (N, 1)
        Counts of the number of times each state is visited
    """
    # initialize the state-action value function and the state counts
    Q = np.zeros((model.num_states, model.num_actions))
    state_counts = np.zeros((model.num_states, 1))

    for i in range(maxeps):

        # epsilon decay from 1 to min_epsilon
        epsilon = max(min_epsilon, 1-(1-min_epsilon)*2/maxeps*i)

        if np.mod(i,1000) == 0:
            print("Running episode %i with epsilon %f" % (i, epsilon))

        # for each new episode, start at the given start state
        state = int(model.start_state_seq)

        # sample first e-greedy action
        action = sample_action(Q, state, model.num_actions, epsilon)

        for j in range(maxiter):
            # initialize p and r
            p, r = 0, np.random.random()

            # sample the next state according to the action and the
            # probability of the transition
            for next_state in range(model.num_states):
                p += model.P[state, next_state, action]
                if r <= p:
                    break

            # epsilon-greedy action selection
            next_action = sample_action(Q, next_state, model.num_actions, epsilon)

            #######################################################################
            # TODO1:                                                              #
            # Update Q function with SARSA backup                                 #
            #######################################################################



            #######################################################################
            #                       END OF YOUR CODE                              #
            #######################################################################

            # End episode is state is a terminal state
            if np.any(state == model.goal_states_seq):
                break

            # count the state visits
            state_counts[state] += 1

            # store the previous state and action
            state = next_state
            action = next_action

    #######################################################################
    # TODO2:                                                              #
    # Determine policy pi from Q function                                 #
    #######################################################################



    #######################################################################
    #                       END OF YOUR CODE                              #
    #######################################################################

    return pi, state_counts

def qlearning(model, alpha=0.5, min_epsilon=0.1, maxiter=100, maxeps=1000):
    """
    Solves the supplied environment using Q-learning.

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.

    alpha : float
        Algorithm learning rate. Defaults to 0.5.

    min_epsilon : float
         epsilon is the probability that a random action is selected.
         epsilon must be in the interval [0,1] where 0 means that the
         action is selected in a completely greedy manner and 1 means
         the action is always selected randomly. Here min_epsilon is
         the minimal epsilon for epsilon decay. In practice, epsilon
         is in the interval [min_epsilon, 1].

    maxiter : int
        The maximum number of iterations to perform per episode.
        Defaults to 100.

    maxeps : int
        The number of episodes to run Q-Learning for.
        Defaults to 1000.

    Returns
    -------
    pi : numpy array of shape (N, 1)
        Optimal policy for the environment where N is the total
        number of states.

    state_counts : numpy array of shape (N, 1)
        Counts of the number of times each state is visited
    """
    # initialize the state-action value function and the state counts
    Q = np.zeros((model.num_states, model.num_actions))
    state_counts = np.zeros((model.num_states, 1))

    for i in range(maxeps):

        # epsilon decay from 1 to min_epsilon
        epsilon = max(min_epsilon, 1-(1-min_epsilon)*2/maxeps*i)
        if np.mod(i,1000) == 0:
            print("Running episode %i with epsilon %f" % (i, epsilon))

        # for each new episode, start at the given start state
        state = int(model.start_state_seq)

        for j in range(maxiter):
            # sample first e-greedy action
            action = sample_action(Q, state, model.num_actions, epsilon)
            # initialize p and r
            p, r = 0, np.random.random()

            # sample the next state according to the action and the
            # probability of the transition
            for next_state in range(model.num_states):
                p += model.P[state, next_state, action]
                if r <= p:
                    break

            #######################################################################
            # TODO3:                                                              #
            # Update Q function with Q-learning backup                            #
            #######################################################################



            #######################################################################
            #                       END OF YOUR CODE                              #
            #######################################################################

            # count the state visits
            state_counts[state] += 1

            #Store the previous state
            state = next_state
            # End episode is state is a terminal state
            if np.any(state == model.goal_states_seq):
                break

    #######################################################################
    # TODO4:                                                              #
    # Determine policy pi from Q function                                 #
    #######################################################################



    #######################################################################
    #                       END OF YOUR CODE                              #
    #######################################################################

    return pi, state_counts

def sample_action(Q, state, num_actions, epsilon):
    """
    Epsilon greedy action selection.

    Parameters
    ----------
    Q : numpy array of shape (N, 1)
        Q function for the environment where N is the total number of states.

    state : int
        The current state.

    num_actions : int
        The number of actions.

    epsilon : float
         Probability that a random action is selected. epsilon must be
         in the interval [0,1] where 0 means that the action is selected
         in a completely greedy manner and 1 means the action is always
         selected randomly.

    Returns
    -------
    action : int
        Number representing the selected action between 0 and num_actions.
    """
    if np.random.random() < epsilon:
        action = np.random.randint(0, num_actions)
    else:
        action = np.argmax(Q[state, :])

    return action
