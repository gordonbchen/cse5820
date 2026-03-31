import torch
import torch.nn.functional as F
import numpy as np
import random
import collections
from agent.agent import Agent
from agent.model import DQN

np.random.seed(1009)
torch.cuda.manual_seed(1009)
torch.manual_seed(1009)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def prepro(o):
    """
    preprocess observations (a stack of 4 last frames)
    """
    o = np.transpose(o, (2,0,1))
    o = np.expand_dims(o, axis=0)
    return o


class Agent_DQN(Agent):

    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN,self).__init__(env)
        
        if args.test_dqn:
            print('loading trained model')
            model = torch.load(args.model_name)
            self.current_net = DQN(84, 84)
            self.current_net.load_state_dict(model['current_net'].state_dict())
            self.hyper_param = args.__dict__
            self.current_net = self.current_net.to(device)
        elif args.train_dqn:
            self.current_net = DQN(84, 84)
            self.target_net = DQN(84, 84)
            self.update_target_net()
            self.step_count = 0
            self.epsilon = 1.0
                
            self.replay_buffer_len = 10000
            self.replay_buffer = collections.deque([], maxlen=self.replay_buffer_len)
            self.optimizer = ['Adam', 'RMSprop', 'SGD']

            self.hyper_param = args.__dict__
            self.training_curve = []

            if self.hyper_param['optim'] in self.optimizer:
                if self.hyper_param['optim'] == 'Adam':
                    self.optimizer = torch.optim.Adam(self.current_net.parameters(), lr = self.hyper_param['learning_rate'], betas = (0.9, 0.999))
                elif self.hyper_param['optim'] == 'RMSprop':
                    self.optimizer = torch.optim.RMSprop(self.current_net.parameters(), lr = self.hyper_param['learning_rate'], alpha = 0.9)
                elif self.hyper_param['optim'] == 'SGD':
                    self.optimizer = torch.optim.SGD(self.current_net.parameters(), lr = self.hyper_param['learning_rate'])
            else:
                print("Unknown Optimizer!")
                exit()
        print(self.current_net)

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        
        pass

    def train(self):
        """
        Implement your training algorithm here
        """
        self.current_net = self.current_net.to(device)
        self.target_net = self.target_net.to(device)
        
        batch_size = self.hyper_param['batch_size']

        #############################################################
        # YOUR CODE HERE                                            #
        # At the end of train, you need to save your model for test #
        #############################################################

































        #############################################################
        # End of YOUR CODE HERE                                     #
        #############################################################


    def policy(self, observation, test=False):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """

        # We only need to use 3 actions. 
        # The index and action is defined below
        #       action 3: left, action 2: right, action 1: stay/fire
        # Note: the returned action is incremented by 1 because action uses 1-based indexing
        # https://gymnasium.farama.org/v0.28.1/environments/atari/breakout/

        if not test:
            q_value = self.current_net(torch.Tensor(observation).to(device))

            if np.random.rand() < self.epsilon:
                action = np.random.randint(3)
                return action+1
            else:
                action = torch.argmax(q_value)
                return action.item()+1
        else:
            observation = prepro(observation)
            q_value = self.current_net(torch.Tensor(observation).to(device))
            return torch.argmax(q_value).item()+1
    
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.current_net.state_dict())
        
    def update_epsilon(self):
        if self.epsilon >= 0.025:
            self.epsilon -= 0.000001

    def save_checkpoint(self, episode = 0):
        check = {'current_net': self.current_net,
                 'target_net': self.target_net,
                 'epsilon_value': self.epsilon,
                 'curve': self.training_curve}
        torch.save(check, self.hyper_param['model_name']+"_"+str(episode))



