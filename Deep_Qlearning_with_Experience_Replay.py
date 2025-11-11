import random
import numpy as np
from collections import deque

import gymnasium as gym
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ------------------ Device ------------------
device = torch.device("cpu")
print(f"Using device: {device}")

# ------------------ Preprocessing ------------------
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (42, 42), interpolation=cv2.INTER_AREA)
    frame = frame / 255.0
    return frame

# ------------------ CNN Q-Network ------------------
class CNN_QNetwork(nn.Module):
    def __init__(self, action_dim):
        super(CNN_QNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 3 * 3, 256),  # correct flattened size
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

# ------------------ DQN Agent ------------------
class DQNAgent:
    def __init__(self, action_dim, gamma=0.99, lr=1e-4, memory_size=50_000):
        self.action_dim = action_dim
        self.gamma = gamma
        self.memory = deque(maxlen=memory_size)

        self.q = CNN_QNetwork(action_dim).to(device)
        self.target_q = CNN_QNetwork(action_dim).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.eval()

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.step_counter = 0

    def store(self, transition):
        self.memory.append(transition)

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q(state_tensor)
        return q_values.argmax().item()

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(states, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        q_values = self.q(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_q(next_states).max(1)[0].unsqueeze(1)
            targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_q.load_state_dict(self.q.state_dict())

# ------------------ Training Loop ------------------
env = gym.make("PongNoFrameskip-v4", render_mode=None)
agent = DQNAgent(action_dim=env.action_space.n)

episodes = 20  # small number for testing
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
batch_size = 32
train_every = 4  # train every 4 steps
target_update_every = 100

for ep in range(episodes):
    frame, _ = env.reset()
    state = preprocess_frame(frame)
    state_stack = np.stack([state] * 2, axis=0)

    total_reward = 0
    done = False
    step = 0
    while not done:
        action = agent.act(state_stack, epsilon)
        next_frame, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = preprocess_frame(next_frame)
        next_state_stack = np.append(state_stack[1:], np.expand_dims(next_state, 0), axis=0)

        agent.store((state_stack, action, reward, next_state_stack, done))

        if step % train_every == 0:
            agent.train(batch_size=batch_size)

        state_stack = next_state_stack
        total_reward += reward
        step += 1
        agent.step_counter += 1

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if agent.step_counter % target_update_every == 0:
            agent.update_target_network()

    print(f"Episode {ep+1} | Reward: {total_reward:.1f} | Epsilon: {epsilon:.3f}")

env.close()
