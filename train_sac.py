import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

# 하이퍼파라미터 
LR_ACTOR = 0.00003      
LR_CRITIC = 0.00003
LR_ALPHA = 0.0003       # 엔트로피 온도 학습률
GAMMA = 0.99            # 미래 보상 할인율
TAU = 0.005             
BUFFER_SIZE = 100000    
BATCH_SIZE = 256        
HIDDEN_DIM = 256       

# 최근 점수 기록
recent_scores = deque(maxlen=20)
SAVE_DIR = "./saved_models_sac"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# 신경망 정의
# 평균과 표준편차를 출력하는 확률적 정책
class SoftActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SoftActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 행동의 평균
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        # 행동의 표준편차
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.rsample() 
        action = torch.tanh(z)
        # Log Probability 계산 (Tanh 보정 포함)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

# Double Q-Learning을 위해 2개 사용
class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(TwinCritic, self).__init__()
        
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        
        # Q2 
        self.r1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.r2 = nn.Linear(hidden_dim, hidden_dim)
        self.r3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.r1(sa))
        q2 = F.relu(self.r2(q2))
        q2 = self.r3(q2)
        return q1, q2

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.FloatTensor(np.array(state)).to(device),
                torch.FloatTensor(np.array(action)).to(device),
                torch.FloatTensor(reward).unsqueeze(1).to(device),
                torch.FloatTensor(np.array(next_state)).to(device),
                torch.FloatTensor(done).unsqueeze(1).to(device))

    def __len__(self):
        return len(self.buffer)
# 유니티 연결
print("유니티 연결 중 - (유니티에서 Play 버튼을 누르세요)")
env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
env.reset()

behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]
state_dim = sum([obs_spec.shape[0] for obs_spec in spec.observation_specs])
action_dim = spec.action_spec.continuous_size 

print(f"State Dim: {state_dim}, Action Dim: {action_dim}")
print("SAC 학습 시작")

actor = SoftActor(state_dim, action_dim, HIDDEN_DIM).to(device)
critic = TwinCritic(state_dim, action_dim, HIDDEN_DIM).to(device)
target_critic = TwinCritic(state_dim, action_dim, HIDDEN_DIM).to(device)
target_critic.load_state_dict(critic.state_dict())

try:
    # 기존에 학습한 최고 모델 불러오기.
    actor.load_state_dict(torch.load(f"{SAVE_DIR}/sac_actor_best.pth"))
    print("✅ 기존 모델 로드 성공")
except:
    print("저장된 모델 없음. 처음부터 시작.")


# 옵티마이저
actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

# 자동 엔트로피 튜닝 - Alpha
target_entropy = -float(action_dim)
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_optimizer = optim.Adam([log_alpha], lr=LR_ALPHA)

memory = ReplayBuffer(BUFFER_SIZE)


# 메인 학습 루프
best_score = -99999.0

for episode in range(1, 3001):
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    obs_list = [decision_steps.obs[i][0] for i in range(len(decision_steps.obs))]
    state = np.concatenate(obs_list, axis=0)

    episode_reward = 0
    
    while True:
        # 행동 결정-(Sampling) 
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            action, _ = actor.sample(state_tensor)
            action = action.cpu().numpy()[0]

        # 유니티 전송 
        action_tuple = ActionTuple()
        action_tuple.add_continuous(np.array([action]))
        env.set_actions(behavior_name, action_tuple)
        env.step()

        decision_steps, terminal_steps = env.get_steps(behavior_name)

        if len(terminal_steps) > 0:
            done = True
            obs_list = [terminal_steps.obs[i][0] for i in range(len(terminal_steps.obs))]
            next_state = np.concatenate(obs_list, axis=0)
            reward = terminal_steps.reward[0]
        else:
            done = False
            obs_list = [decision_steps.obs[i][0] for i in range(len(decision_steps.obs))]
            next_state = np.concatenate(obs_list, axis=0)
            reward = decision_steps.reward[0]

        memory.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        # 학습
        if len(memory) > BATCH_SIZE:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

            with torch.no_grad():
                next_actions, next_log_probs = actor.sample(next_states)
                target_q1, target_q2 = target_critic(next_states, next_actions)
                target_min_q = torch.min(target_q1, target_q2)
                
                # entropy regularization  
                alpha = log_alpha.exp()
                target_q = rewards + GAMMA * (1 - dones) * (target_min_q - alpha * next_log_probs)

            current_q1, current_q2 = critic(states, actions)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor 
            new_actions, log_probs = actor.sample(states)
            q1_new, q2_new = critic(states, new_actions)
            min_q_new = torch.min(q1_new, q2_new)

            actor_loss = (alpha * log_probs - min_q_new).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Alpha 자동 튜닝 
            alpha_loss = -(log_alpha * (log_probs + target_entropy).detach()).mean()

            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            with torch.no_grad():
                log_alpha.data.clamp_(max=np.log(0.01))

            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        if done:
            break

    recent_scores.append(episode_reward)
    avg_score = np.mean(recent_scores)
    
    # 학습된 Alpha 값 확인
    current_alpha = log_alpha.exp().item()

    print(f"Ep {episode} | Score: {episode_reward:.2f} (Avg: {avg_score:.1f}) | Alpha: {current_alpha:.4f}")

    if episode_reward > best_score:
        best_score = episode_reward
        torch.save(actor.state_dict(), f"{SAVE_DIR}/sac_actor_best.pth")
        print(f" 최고 기록 ({best_score:.2f})")

env.close()