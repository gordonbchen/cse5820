import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym


def build_mlp(sizes: list[int]):
    layers = []
    for i in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i != len(sizes) - 2: layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# Hyperparam.
N_ENVS = 1
N_EPOCHS = 256
N_STEPS = 1024
DEVICE = "cpu"
LOG_STEPS = 10
VAL_STEPS = 1024

HIDDEN_SIZE = 32

POLICY_LR = 1e-3
POLICY_OPTIM_STEPS = 2
CRITIC_LR = 1e-3
CRITIC_OPTIM_STEPS = 4

DISCOUNT_GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2


# Cartpole: http://gymnasium.farama.org/environments/classic_control/cart_pole/
# Same step reset: https://farama.org/Vector-Autoreset-Mode
envs = gym.vector.SyncVectorEnv(
   [lambda : gym.make("CartPole-v1") for _ in range(N_ENVS)],
    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
)
N_OBS = envs.single_observation_space.shape[0]
N_ACTIONS = int(envs.single_action_space.n)
print(f"obs={N_OBS}, actions={N_ACTIONS}")


# Build models.
policy = build_mlp([N_OBS, HIDDEN_SIZE, N_ACTIONS])
policy.to(DEVICE)
print(policy)

critic = build_mlp([N_OBS, HIDDEN_SIZE, 1])
critic.to(DEVICE)
print(critic)

policy_optim = Adam(policy.parameters(), lr=POLICY_LR)
critic_optim = Adam(critic.parameters(), lr=CRITIC_LR)


# Storage buffers.
obss = torch.zeros((N_STEPS+1, N_ENVS, N_OBS), dtype=torch.float32, device=DEVICE)
dones = torch.zeros((N_STEPS+1, N_ENVS), dtype=torch.float32, device=DEVICE)
actions = torch.zeros((N_STEPS, N_ENVS), dtype=torch.int64, device=DEVICE)
rewards = torch.zeros((N_STEPS, N_ENVS), dtype=torch.float32, device=DEVICE)
advantages = torch.zeros((N_STEPS, N_ENVS), dtype=torch.float32, device=DEVICE)
old_log_probs = torch.zeros((N_STEPS, N_ENVS), dtype=torch.float32, device=DEVICE)

obs, _ = envs.reset()
obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
done = torch.zeros(N_ENVS, dtype=torch.float32, device=DEVICE)

log = SummaryWriter()
for epoch in range(N_EPOCHS):
    # obs and done continue from prev epoch
    obss[0] = obs
    dones[0] = done
    for t in range(N_STEPS):
        with torch.no_grad():
            logits = policy(obs)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        old_log_probs[t] = action_dist.log_prob(action)
        actions[t] = action

        obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
        rewards[t] = torch.tensor(reward, dtype=torch.float32, device=DEVICE)

        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        done = torch.tensor(terminated | truncated, dtype=torch.float32, device=DEVICE)
        obss[t+1] = obs
        dones[t+1] = done

    # GAE.
    with torch.no_grad():
        values = critic(obss).squeeze(-1)
        gae_accum = torch.zeros(N_ENVS, dtype=torch.float32, device=DEVICE)
        for t in reversed(range(N_STEPS)):
            # If not done, bootstrap w/ next value.
            next_nonterminal = 1 - dones[t+1]
            delta = rewards[t] + DISCOUNT_GAMMA * values[t+1] * next_nonterminal - values[t]
            advantages[t] = gae_accum = delta + DISCOUNT_GAMMA * GAE_LAMBDA * gae_accum * next_nonterminal
        value_targets = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Optimize policy.
    for _ in range(POLICY_OPTIM_STEPS):
        log_probs = Categorical(logits=policy(obss[:-1])).log_prob(actions)
        ratios = (log_probs - old_log_probs).exp()
        policy_loss = -(torch.minimum(ratios * advantages, ratios.clip(1-CLIP_EPS, 1+CLIP_EPS) * advantages)).mean()
        policy_optim.zero_grad(set_to_none=True)
        policy_loss.backward()
        policy_optim.step()

    # Optimize critic.
    for _ in range(CRITIC_OPTIM_STEPS):
        critic_loss = F.mse_loss(critic(obss[:-1]).squeeze(-1), value_targets)
        critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_optim.step()

    # Log.
    if (epoch % LOG_STEPS == 0) or (epoch == N_EPOCHS - 1): 
        policy.eval()
        total_rewards = 0
        cum_reward = torch.zeros(N_ENVS, dtype=torch.float32, device=DEVICE)
        full_episodes = 0

        obs, _ = envs.reset()
        done.zero_()
        for _ in range(VAL_STEPS):
            obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                action = policy(obs).argmax(dim=-1)

            obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
            reward = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
            done = torch.tensor(terminated | truncated, dtype=torch.float32, device=DEVICE)

            cum_reward += reward
            total_rewards += (cum_reward @ done).item()
            cum_reward *= 1-done
            full_episodes += done.sum().item()

        # Reset for training.
        policy.train()
        obs, _ = envs.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        done.zero_()

        full_reward_mean = total_rewards / full_episodes
        print(f"epoch={epoch} mean_reward={full_reward_mean}")
        log.add_scalar("mean_reward", full_reward_mean, epoch)
        log.add_scalar("full_episodes", full_episodes, epoch)
        log.add_scalar("loss/policy", policy_loss.item(), epoch)
        log.add_scalar("loss/critic", critic_loss.item(), epoch)
log.close()