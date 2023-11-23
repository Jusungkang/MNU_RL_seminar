##
## 작성일 : 20231123
## 작성자 : Jusung Kang
## 목적 : MountainCar_w_DQN 예제 셈플 코드 주석 작성 (다른 ENV 활용 목적)
## 참조 URL : Pytorch 강화학습 튜토리얼 (https://tutorials.pytorch.kr/intermediate/reinforcement_q_learning.html)
## MDPs 정의 URL : https://gymnasium.farama.org/environments/classic_control/mountain_car/
##

## open AI Gym 포함 필요 library 호출
import gymnasium as gym

import random, math, matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


## Duration, 즉 Rewards 값을 Figure 그래프화 위함.
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # 도표가 업데이트되도록 잠시 멈춤
    plt.pause(0.001)

    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

## Memory 속 Episode samples 를 DQN에 입/출력하기 위한 형태로 재 정렬하여 DQN 학습 진행.
## Data structure 구성 형태는 다양하게 가능.
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). 이것은 batch-array의 Transitions을 Transition의 batch-arrays로
    # 전환합니다.
    batch = Transition(*zip(*transitions))

    # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
    # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
    # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
    # max(1)[0]으로 최고의 보상을 선택하십시오.
    # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber 손실 계산
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    # 변화도 클리핑 바꿔치기
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

## Episode로부터 나오는 학습 셈플들을 관리하기 위함.
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

## DQN 정의부. 2 layer FC network 사용.
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # 최적화 중에 다음 행동을 결정하기 위해서 하나의 요소 또는 배치를 이용해 호촐됩니다.
    # ([[left0exp,right0exp]...]) 를 반환합니다.
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

## Action Space 함수
## Eps 에 따라 DQN 결과값 혹은 random action 값을 출력해줌을 확인 가능.
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
            # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

## MountainCar Envrionment 호출 함수
env = gym.make('MountainCar-v0', render_mode="human") # MountainCar 환경 생성


# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# GPU를 사용할 경우를 위한 torch device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay memory 저장을 위한 데이터 구조 설정
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 128        # BATCH_SIZE는 리플레이 버퍼에서 샘플링된 트랜지션의 수입니다.
GAMMA = 0.99            # GAMMA는 이전 섹션에서 언급한 할인 계수입니다.
EPS_START = 0.9         # EPS_START는 엡실론의 시작 값입니다.
EPS_END = 0.05          # EPS_END는 엡실론의 최종 값입니다.
EPS_DECAY = 1000        # EPS_DECAY는 엡실론의 지수 감쇠(exponential decay) 속도 제어하며, 높을수록 감쇠 속도가 느립니다.
TAU = 0.005             # TAU는 목표 네트워크의 업데이트 속도입니다.
LR = 1e-4               # LR은 ``AdamW`` 옵티마이저의 학습율(learning rate)입니다.


# Env. 내 가능한 Action Space 크기를 확인하기 위함. DQN 마지막 layer 에서의 크기로 연결
n_actions = env.action_space.n

## Env. 초기화
state, info = env.reset()

# Env. 내 측정되는 Observation space 의 크기를 확인하기 위함. DQN 첫 layer 에서의 입력 크기로 연결
n_observations = len(state)

## Main & Target DQN 정의
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

episode_durations = []

# GPU 사용에 따른 Episodes 정의. CPU가 아무래도 느리다보니....
if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

# RL Training 시작
for i_episode in range(num_episodes):

    # Env. 초기화
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    for t in count():
        # State 입력에 따른 Action 선택
        action = select_action(state)
        # Action 결과에 따른 다음 State, Observation Reword 등의 결과 확인.
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Episode 셈플을 Replay memory에 저장
        memory.push(state, action, next_state, reward)

        # 다음 state로 이동
        state = next_state

        # Policy Network 최적화 한단계 수행
        optimize_model()

        # Policy 학습 결과에 따른, Target Network 소프트 업데이트
        # 보통 일정 episode (e.g. 100) 동안에 한번씩 발생시킴.
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
