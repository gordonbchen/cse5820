from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from agent.agent import Agent
import numpy as np
import time

np.random.seed(1009)
torch.cuda.manual_seed(1009)
torch.manual_seed(1009)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def prepro(o, image_size=[80, 80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    """
    # Crop.
    o = o[33:195]

    # Normalize and grayscale.
    o = torch.tensor(o, dtype=torch.float32, device=device) / 255.0
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]

    # Resize and delete scoreboard.
    y = y.view(1, 1, *y.shape)
    resized = F.interpolate(y, size=image_size, mode="area")

    # import matplotlib.pyplot as plt
    # plt.imshow(resized[0, :, :, None].cpu().numpy(), cmap="grey")
    # plt.show()
    return resized


class Agent_PG(Agent):

    def __init__(self, env, args):
        super(Agent_PG, self).__init__(env)
        
        if args.test_pg:
            print('loading trained model')
            self.model = torch.load(args.model_name, weights_only=False)
            self.hyper_param = args.__dict__
            self.last_frame = None

        elif args.train_pg:
            self.hyper_param = args.__dict__
            self.argument = args
            self.episode_rewards = []
            self.last_frame = None
            
            # initial model
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(80*80*1, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ).to(device)

            self.critic = nn.Sequential(
                nn.Flatten(),
                nn.Linear(80*80*1, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            ).to(device)
                
            # initial optimizer
            if self.hyper_param['optim'] == 'Adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyper_param['learning_rate'], betas=(0.9, 0.999))
                self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.hyper_param['learning_rate'], betas=(0.9, 0.999))
            elif self.hyper_param['optim'] == 'RMSprop':
                self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr = self.hyper_param['learning_rate'], alpha = 0.9)
                self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=self.hyper_param['learning_rate'], alpha=0.9)
            elif self.hyper_param['optim'] == 'SGD':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.hyper_param['learning_rate'])
                self.critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=self.hyper_param['learning_rate'])
            else:
                print("Unknown Optimizer!")
                exit()

    def init_game_setting(self):
        """
        Testing function will call this function at the beginning of new game
        """
        self.last_frame = None

    # @profile
    def train(self):
        print(self.hyper_param)
        GAMMA = self.hyper_param['gamma']
        BATCH_SIZE = self.hyper_param['batch_size']
        N_EPISODES = self.hyper_param['episode']
        EPS = 1e-9

        save_dir = Path("runs") / self.hyper_param['model_name']
        save_dir.mkdir(parents=True, exist_ok=True)

        #############################################################
        # YOUR CODE HERE                                            #
        # At the end of train, you need to save your model for test #
        #############################################################
        total_time = 0
        best_reward = -float("inf")
        for episode in range(N_EPISODES):
            t0 = time.time()
            log_probs, rewards, values = [], [], []
            self.init_game_setting()
            obs = self.env.reset()
            while True:
                # Preprocess and diff frames.
                cur_obs = prepro(obs)
                if self.last_frame is None:  # no motion, velocity = 0s.
                    obs = torch.zeros_like(cur_obs)
                else:
                    obs = cur_obs - self.last_frame
                self.last_frame = cur_obs

                action, prob3 = self.policy(obs, test=False)
                prob = prob3 if action == 3 else 1 - prob3
                log_probs.append((prob + EPS).log())

                values.append(self.critic(obs))

                obs, reward, terminated, _ = self.env.step(action)
                rewards.append(reward)

                if terminated:
                    break

            # Calculate total rewards.
            total_reward = sum(rewards)
            self.episode_rewards.append(total_reward)

            # Rewards to go, don't carry over points (b/c pong resets).
            for i in range(len(rewards)-2, -1, -1):
                if abs(rewards[i]) < 0.5:  # nonzero reward, abs < for fp comp.
                    rewards[i] += GAMMA * rewards[i+1]
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

            # Compute advantages.
            values = torch.cat(values, dim=1)[0]
            advantages = rewards - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

            # Optimize policy.
            log_probs = torch.cat(log_probs, dim=1)[0]
            policy_loss = (-log_probs * advantages).sum()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            # Optimize critic.
            critic_loss = F.mse_loss(values, rewards)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Log.
            dt = time.time() - t0
            total_time += dt
            rem_time = ((N_EPISODES - episode) - 1) * total_time / ((episode + 1) * 60)
            print(f"{episode}: reward={total_reward} steps={len(log_probs)}", end=" ")
            print(f"policy_loss={policy_loss.item():.5f}, critic_loss={critic_loss.item():.5f}", end=" ")
            print(f"dt={dt:.3f} rem_time={rem_time:.3f}")

            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(self.model, save_dir / "best.pt")

        torch.save(self.model, save_dir / "final.pt")
        #############################################################
        # END OF YOUR CODE HERE                                     #
        #############################################################


        

    def policy(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: torch.Tensor
                preprocessed frame (pre-diff)

        Return:
            action: int
                the predicted action from trained model
        """
        # We only need two actions. The index and action is defined below 
        #    action 2: up, action 3: down
        # https://gymnasium.farama.org/v0.28.0/environments/atari/pong/

        if not test:
            output = self.model(observation)
            probability = output[0,0]
            if np.random.rand() < probability.item():
                action = 3
            else:
                action = 2
            return action, output
        elif test:
            if type(self.last_frame) == type(None):
                self.last_frame = prepro(observation)
                observation = torch.zeros_like(self.last_frame)
            else:
                o = prepro(observation)
                observation = o - self.last_frame
                self.last_frame = o
            output = self.model(observation)
            probability = output[0,0]
            if np.random.rand() < probability.item():
                action = 3
            else:
                action = 2
            return action
