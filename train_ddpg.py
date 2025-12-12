import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


# 하이퍼파라미터
LR_ACTOR = 0.00005     # Actor 학습률
LR_CRITIC = 0.001       # Critic 학습률
GAMMA = 0.995           # 미래 보상 할인율
TAU = 0.005            
BUFFER_SIZE = 100000     
BATCH_SIZE = 128         
NOISE_STD = 0.4        # 노이즈 
DECAY_RATE = 0.999    # 노이즈 감소율

# 최근 점수 평균 계산용
recent_scores = deque(maxlen=20)

# 모델 저장 경로
SAVE_DIR = "./saved_models_ddpg"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU:", device)

# NN Define
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.tanh = nn.Tanh() # 출력 범위를 -1.0 ~ 1.0 으로 제한

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.tanh(self.fc3(x))

# Critic: 상태(S)와 행동(A)을 보고 -> 점수(Q)를 매김
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # 상태와 행동을 입력으로 합쳐서 받음
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1) # 점수 1개 출력

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1) # 데이터 합치기
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
                torch.FloatTensor(np.array(action)).to(device),
                torch.FloatTensor(reward).unsqueeze(1).to(device),
                torch.FloatTensor(np.array(next_state)).to(device),
                torch.FloatTensor(done).unsqueeze(1).to(device))

    def __len__(self):
        return len(self.buffer)

# ==========================================
# [4] 유니티 환경 연결 및 초기화
# ==========================================
print("유니티 연결 중... (에디터에서 Play 버튼을 누르세요)")
print("5004번 포트로 연결 시도 중...")
env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
env.reset()

# 행동 이름 가져오기
behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]

# [중요] 입력(State) 크기 자동 계산 (위치정보 + 레이저센서 등 모두 합침)
state_dim = sum([obs_spec.shape[0] for obs_spec in spec.observation_specs])

# 출력(Action) 크기 가져오기 (Continuous)
action_dim = spec.action_spec.continuous_size 

print(f"State Dim: {state_dim}, Action Dim: {action_dim}")
print("DDPG 학습 시작!")

# ==========================================
# [5] 모델 및 최적화기 생성
# ==========================================
# 실제 학습할 네트워크
actor = Actor(state_dim, action_dim).to(device)
critic = Critic(state_dim, action_dim).to(device)

# 정답 계산용 타깃 네트워크 (천천히 업데이트됨)
target_actor = Actor(state_dim, action_dim).to(device)
target_critic = Critic(state_dim, action_dim).to(device)

# 가중치 복사
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

memory = ReplayBuffer(BUFFER_SIZE)
noise_std = NOISE_STD # 현재 노이즈 레벨

# ==========================================
# [6] 메인 학습 루프
# ==========================================
best_score = -99999.0

for episode in range(1, 2001): # 2000 에피소드 진행
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    
    # [데이터 병합] 여러 센서 데이터를 하나로 합침
    obs_list = [decision_steps.obs[i][0] for i in range(len(decision_steps.obs))]
    state = np.concatenate(obs_list, axis=0)

    episode_reward = 0
    step_count = 0

    while True:
        # --- 1. 행동 결정 (Actor + Noise) ---
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = actor(state_tensor).cpu().numpy()[0]

        # 탐험을 위해 노이즈 추가
        noise = np.random.normal(0, noise_std, size=action_dim)
        action = action + noise
        action = np.clip(action, -1.0, 1.0) # -1 ~ 1 범위 유지

        # --- 2. 유니티로 전송 ---
        action_tuple = ActionTuple()
        action_tuple.add_continuous(np.array([action]))
        env.set_actions(behavior_name, action_tuple)
        env.step()

        # --- 3. 결과 받기 ---
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        if len(terminal_steps) > 0: # 에피소드 종료
            done = True
            # 터미널 스텝의 관측 정보 가져오기
            obs_list = [terminal_steps.obs[i][0] for i in range(len(terminal_steps.obs))]
            next_state = np.concatenate(obs_list, axis=0)
            reward = terminal_steps.reward[0]
        else: # 계속 진행
            done = False
            # 디시전 스텝의 관측 정보 가져오기
            obs_list = [decision_steps.obs[i][0] for i in range(len(decision_steps.obs))]
            next_state = np.concatenate(obs_list, axis=0)
            reward = decision_steps.reward[0]

        # --- 4. 저장 ---
        memory.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        step_count += 1

        # --- 5. 학습 (Update) ---
        if len(memory) > BATCH_SIZE:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

            # [Critic 업데이트]
            with torch.no_grad():
                # 타깃 Actor가 다음 행동을 선택
                next_actions = target_actor(next_states)
                # 타깃 Critic이 그 행동의 점수를 평가
                target_q = target_critic(next_states, next_actions)
                # 정답 Q값 계산 (보상 + 할인된 미래 점수)
                expected_q = rewards + (GAMMA * target_q * (1 - dones))

            # 현재 Critic의 예측값
            current_q = critic(states, actions)
            critic_loss = nn.MSELoss()(current_q, expected_q)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # [Actor 업데이트]
            # Critic을 속여서 높은 점수를 받는 행동을 하도록 학습
            actor_loss = -critic(states, actor(states)).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # [Soft Update] 타깃 네트워크를 조금씩 메인 네트워크와 동기화
            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
            
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

        if done:
            break
    
    # 1. 최근 점수 기록
    recent_scores.append(episode_reward)
    avg_score = np.mean(recent_scores)

    # 2. 실력에 따른 감소율(Decay Rate) 결정
    if avg_score >= 50.0:
        current_decay = 0.9      # 고수: 노이즈를 10%씩 팍팍 깎음 (졸업 준비)
    elif avg_score >= 25.0:
        current_decay = 0.99     # 중수: 1%씩 깎음 (정석)
    else:
        current_decay = 0.999    # 초보: 0.1%씩 깎음 (아직 탐험이 더 필요함)

    # 3. 노이즈 적용 
    noise_std = max(0.15, noise_std * current_decay)

    print(f"Episode {episode} | Reward: {episode_reward:.2f} (Avg: {avg_score:.1f}) | Noise: {noise_std:.3f}")

    # 모델 저장 (최고 기록 갱신 시)
    if episode_reward > best_score:
        best_score = episode_reward
        torch.save(actor.state_dict(), f"{SAVE_DIR}/ddpg_actor_best.pth")
        torch.save(critic.state_dict(), f"{SAVE_DIR}/ddpg_critic_best.pth")
        print(f"★ 최고 기록 갱신 모델 저장됨 ({best_score:.2f})")

env.close()
print("학습 종료")