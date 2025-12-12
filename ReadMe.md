# Unity 강화학습 프로젝트: 장애물 회피 자율주행 에이전트

## 프로젝트 개요
Unity ML-Agents를 활용하여 장애물을 회피하며 목표 지점까지 이동하는 자율주행 에이전트를 강화학습으로 학습시키는 프로젝트입니다.

## 주요 목표
- 무작위로 배치된 장애물 환경에서 목표 지점(Target) 도달
- 벽과의 충돌 최소화
- 효율적인 경로 탐색 및 주행 전략 학습

## 프로젝트 구조
```
Project
├── Unity Scripts
│   ├── CarAgent.cs          # ML-Agents 에이전트 (관측, 보상, 행동)
│   └── Controller.cs        # 차량 물리 제어 (조향, 가속)
│
└── Python Training Scripts
    ├── DQN.py              # Deep Q-Network (이산 행동)
    ├── train_ddpg.py       # DDPG (연속 행동)
    ├── train_sac.py        # SAC (연속 행동, 최종 채택)
    └── run_inference.py    # 학습된 모델 추론 실행
```

## 구현된 강화학습 알고리즘

### 1. DQN (Deep Q-Network)
- **타입**: Value-based, 이산 행동 공간
- **특징**: 5가지 행동 중 선택 (전진, 후진, 좌회전 등)
- **한계**: 연속적인 조향 제어 불가

### 2. DDPG (Deep Deterministic Policy Gradient)
- **타입**: Actor-Critic, 연속 행동 공간
- **특징**: 부드러운 조향 및 가속 제어
- **노이즈**: Gaussian Noise로 탐험

### 3. SAC (Soft Actor-Critic) **최종 채택**
- **타입**: Actor-Critic, 최대 엔트로피 강화학습
- **장점**: 
  - 안정적인 학습
  - 자동 탐험-활용 균형 조절
  - 과적합 방지

## Unity 환경 설정

### CarAgent.cs 주요 기능

#### 관측 (Observation)
```csharp
- 가장 가까운 타깃 방향 (로컬 좌표계, 정규화)
- 타깃까지의 거리
- 차량 속도 (로컬 좌표계)
```

#### 행동 (Action)
```csharp
- 조향 입력: -1.0 ~ 1.0 (좌회전 ~ 우회전)
- 모터 입력: -1.0 ~ 1.0 (후진 ~ 전진)
```

#### 보상 시스템 (Reward)
| 상황 | 보상 | 목적 |
|------|------|------|
| 타깃 도달 | +15.0 | 목표 달성 강화 |
| 벽 충돌 | -1.0 ~ -10.0 | 충돌 방지 (속도 비례) |
| 갇힘 감지 (3초) | -5.0 | 교착 상태 탈출 유도 |
| 타깃 접근 | +거리 변화량 | 효율적 경로 학습 |
| 제자리 회전 | -0.1 | 뱅뱅이 방지 |
| 벽 앞 급정거 실패 | -0.05 × 초과속도 | 안전 속도 학습 |
| 매 스텝 | -0.005 | 빠른 해결 유도 |

#### 비상 탈출 메커니즘 (Panic Mode)
```csharp
// 3초간 1.5m 미만 이동 시 발동
- 강제 후진 2초
- 좌우 장애물 감지 후 회피 방향 결정
- 신경망이 후진 학습 유도 (보상 제공)
```

### Controller.cs - 차량 물리
```csharp
- 4륜 독립 제어 (WheelCollider)
- 안티롤 바 시뮬레이션 (전복 방지)
- 다운포스 적용 (고속 안정성)
- 무게중심 조정
```

## Python 학습 스크립트

### train_sac.py 

#### 하이퍼파라미터
```python
LR_ACTOR = 0.00003          # Actor 학습률
LR_CRITIC = 0.00003         # Critic 학습률
GAMMA = 0.99                # 할인율
BATCH_SIZE = 256            # 배치 크기
BUFFER_SIZE = 100000        # 리플레이 버퍼
HIDDEN_DIM = 256            # 은닉층 차원
```

#### 실행 방법
```bash
# 1. Unity 에디터에서 Play 버튼 대기
python train_sac.py

# 2. 학습 모니터링
# 터미널에서 에피소드별 점수 확인
# Ep 150 | Score: 28.45 (Avg: 22.3) | Alpha: 0.0082
```

#### 모델 저장
```python
# 최고 점수 경신 시 자동 저장
./saved_models_sac/sac_actor_best.pth
```

### run_inference.py - 학습된 모델 테스트
```bash
python run_inference.py

# 학습된 에이전트의 실시간 플레이 관찰
# Ctrl+C로 종료
```

## 📊 학습 결과 분석

### 성능 지표
```
초기 (1-50 에피소드): -5 ~ 10점 (랜덤 행동)
중기 (100-500): 15 ~ 30점 (타깃 1-2개 도달)
후기 (1000+): 40 ~ 60점 (전체 타깃 안정적 수집)
```

### 학습 곡선 특징
1. **초반 (0-100 에피소드)**: 급격한 개선 (충돌 빈도 감소)
2. **중반 (100-500)**: 완만한 상승 (경로 최적화)
3. **후반 (500+)**: 수렴 (안정적 성능)

## 🚀 시작하기

### 필수 요구사항
```bash
# Python 환경
Python 3.8+
PyTorch 1.10+
ml-agents==0.30.0

# Unity 환경
Unity 2022.3.62f2 
ML-Agents Package 3.0.0
```

### 설치 및 실행
```bash
# 1. Python 패키지 설치
pip install torch mlagents

# 2. Unity 프로젝트 열기
# - Unity Hub에서 프로젝트 로드
# - ML-Agents Package 확인

# 3. 학습 시작
python train_sac.py

# 4. 추론 (학습 후)
python run_inference.py
```

## 커스터마이징

### 난이도 조정
```csharp
// CarAgent.cs - OnEpisodeBegin()
public float radius = 50.0f;  // 맵 크기 (큰수록 어려움)
```

### 보상 튜닝
```csharp
// CarAgent.cs - OnTriggerEnter()
AddReward(15.0f);  // 타깃 보상 조정
```

### 학습 속도 조절
```csharp
// CarAgent.cs - Start()
Time.timeScale = 1.5f;  // 시뮬레이션 배속 (최대 20x 권장)
```


## 📈 개선 방향

- [ ] PPO 알고리즘 구현 비교
- [ ] Curriculum Learning (단계별 난이도 증가)
- [ ] 다중 에이전트 협력 학습
- [ ] LSTM 추가 (시계열 정보 활용)
- [ ] Hindsight Experience Replay

## 라이센스

MIT License

## 제작 정보

**제작**: 2025년 강화학습 프로젝트  
**알고리즘**: DQN, DDPG, SAC  
**환경**: Unity ML-Agents + PyTorch