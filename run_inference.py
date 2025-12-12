import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


MODEL_PATH = "./saved_models_sac/sac_actor_best.pth" 
HIDDEN_DIM = 256
device = torch.device("cpu")


class SoftActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SoftActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
      
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        return mean 

   
    def get_action(self, state):
        mean = self.forward(state)

        return torch.tanh(mean)


print("유니티 연결 중... (에디터에서 Play 버튼을 누르세요)")
env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
env.reset()

# 행동 이름 및 차원 자동 가져오기
behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]
state_dim = sum([obs_spec.shape[0] for obs_spec in spec.observation_specs])
action_dim = spec.action_spec.continuous_size 

print(f"State Dim: {state_dim}, Action Dim: {action_dim}")
print("추론 모드 시작")

# 모델 로드
actor = SoftActor(state_dim, action_dim, HIDDEN_DIM).to(device)

try:
    actor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("모델 로드 성공")
except FileNotFoundError:
    print(f"오류: 모델 파일이 없습니다 경로를 확인하세요: {MODEL_PATH}")
    env.close()
    exit()

actor.eval() 

try:
    while True:
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        episode_reward = 0
        step_count = 0

        while True:
            
            if len(decision_steps) == 0:
        
                break
                
            obs_list = [decision_steps.obs[i][0] for i in range(len(decision_steps.obs))]
            state = np.concatenate(obs_list, axis=0)

         
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
             
                action = actor.get_action(state_tensor).cpu().numpy()[0]

           
            action_tuple = ActionTuple()
            action_tuple.add_continuous(np.array([action]))
            env.set_actions(behavior_name, action_tuple)
            
            
            env.step()

 
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(terminal_steps) > 0:
                reward = terminal_steps.reward[0]
                episode_reward += reward
                print(f"Episode 종료 | 점수: {episode_reward:.2f}")
                break 
            
      
            reward = decision_steps.reward[0]
            episode_reward += reward
            step_count += 1

except KeyboardInterrupt:
    print("\n종료 요청됨.")
finally:
    env.close()
    print("유니티 연결 종료.")