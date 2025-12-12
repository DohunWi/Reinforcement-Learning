import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

# ==========================================
# [1] 하이퍼파라미터 설정
# ==========================================
LEARNING_RATE = 0.0005
GAMMA = 0.99          # 미래 보상 할인율
BUFFER_SIZE = 10000   # 리플레이 버퍼 크기
BATCH_SIZE = 64       # 한 번에 학습할 데이터 양
EPSILON_START = 1.0   # 초기 탐험 확률 (100% 랜덤)
EPSILON_END = 0.05    # 최소 탐험 확률 (5%)
EPSILON_DECAY = 0.95 # 탐험 감소 비율
TARGET_UPDATE = 100   # 타깃 네트워크 업데이트 주기

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# [2] Q-Network (두뇌)
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size) # 출력: 각 행동별 가치(Q값)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ==========================================
# [3] 리플레이 버퍼 (기억 저장소)
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.FloatTensor(np.array(state)).to(device),
                torch.LongTensor(action).unsqueeze(1).to(device),
                torch.FloatTensor(reward).unsqueeze(1).to(device),
                torch.FloatTensor(np.array(next_state)).to(device),
                torch.FloatTensor(done).unsqueeze(1).to(device))

    def __len__(self):
        return len(self.buffer)

# ==========================================
# [4] 유니티 환경 연결
# ==========================================
print('Press <Play> button in Unity')

env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
env.reset()
behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]

state_dim = sum([obs_spec.shape[0] for obs_spec in spec.observation_specs])
action_dim = spec.action_spec.discrete_branches[0] # 출력 크기 (5)

print(f"State: {state_dim}, Action: {action_dim}")

# 네트워크 & 최적화기 초기화
policy_net = QNetwork(state_dim, action_dim).to(device) # 행동 결정용
target_net = QNetwork(state_dim, action_dim).to(device) # 정답 계산용
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer(BUFFER_SIZE)

# ==========================================
# [5] 학습 루프 (Main Loop)
# ==========================================
epsilon = EPSILON_START
total_step = 0

print("DQN 학습 시작! 유니티 화면을 보세요.")

for episode in range(1000): # 1000판 진행
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    obs_list = [decision_steps.obs[i][0] for i in range(len(decision_steps.obs))]
    state = np.concatenate(obs_list, axis=0)
    episode_reward = 0

    while True:
        # 1. 행동 결정 (Epsilon-Greedy)
        if random.random() < epsilon:
            action = random.randint(0, action_dim - 1) # 랜덤 행동
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item() # Q값이 가장 높은 행동

        # 2. 유니티에 행동 전송
        action_tuple = ActionTuple()
        action_tuple.add_discrete(np.array([[action]]))
        env.set_actions(behavior_name, action_tuple)
        env.step()

        # 3. 결과 받기 (Next State, Reward, Done)
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        done = False
        if len(terminal_steps) > 0: # 에피소드 종료 (죽거나 성공)
            done = True
            obs_list = [terminal_steps.obs[i][0] for i in range(len(terminal_steps.obs))]
            next_state = np.concatenate(obs_list, axis=0)
            # next_state = terminal_steps.obs[0][0]
            reward = terminal_steps.reward[0]
            # 터미널 스텝 처리 후 루프 종료 준비
        else: # 계속 진행 중
            obs_list = [decision_steps.obs[i][0] for i in range(len(decision_steps.obs))]
            next_state = np.concatenate(obs_list, axis=0)
            # next_state = decision_steps.obs[0][0]
            reward = decision_steps.reward[0]

        # 4. 기억 저장 (Replay Buffer)
        memory.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        total_step += 1

        # 5. 학습 (Experience Replay) - 데이터가 좀 쌓이면 시작
        if len(memory) > BATCH_SIZE:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

            # Q값 계산 (예측값)
            current_q = policy_net(states).gather(1, actions)
            
            # 타깃 Q값 계산 (정답값: r + gamma * max Q')
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                expected_q = rewards + (GAMMA * max_next_q * (1 - dones))

            # Loss 계산 및 업데이트 (MSE Loss)
            loss = nn.MSELoss()(current_q, expected_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 타깃 네트워크 주기적 업데이트
            if total_step % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if done:
            break

    # 에피소드 종료 후 처리
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY) # 탐험 확률 감소
    print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.2f}")

env.close()