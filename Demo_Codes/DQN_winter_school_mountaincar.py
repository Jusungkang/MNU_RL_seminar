import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from collections import deque
import gym

class DQN:
    def __init__(self, session, input_size, output_size, name="main"): # DQN class 초기화 함수
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=10, l_rate=1e-1):
        with tf.variable_scope(self.net_name): # 네트워크의 네임스페이스를 정의
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            W1 = tf.get_variable("W1", shape=[self.input_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer()) # 입력에서 은닉층으로 연결된 가중치
            layer1 = tf.nn.tanh(tf.matmul(self._X, W1)) # 활성 함수
            W2 = tf.get_variable("W2", shape=[h_size, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer()) # 은닉층에서 출력으로 연결된 가중치
            self._Qpred = tf.matmul(layer1, W2) # Q 함수의 선형 회귀 추정을 위한 가설(hypothesis)수립
            self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32) # 출력(action)을 위한 변수
            self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred)) # 손실 함수(Loss function)
            self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss) # 최적화 (학습)

    def update(self, x_stack, y_stack): # 네트워크 업데이트 함수
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})

    def predict(self, state): # 학습된 DQN Network를 통한 Q 함수의 선형 회귀 추정 함수
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)    # 학습시 활용할 Input Buffer 선언
    y_stack = np.empty(0).reshape(0, output_size)   # 학습시 활용할 Output Buffer 선언

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)  # main DQN을 이용한 Q 예측

        if done: # 에피소드 종료시
            Q[0, action] = reward
        else: # 에피소드 미 종료시, target DQN을 이용한 다음 상태에 대한 Q 예측 (학습시 True로 이용)
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

        x_stack = np.vstack([x_stack, state])   # 상태 정보를 Input buffer에
        y_stack = np.vstack([y_stack, Q])       # 예측된 Q를 Output buffer에 저장

    return mainDQN.update(x_stack, y_stack) # main Network 학습 진행

def network_update(*, dest_scope_name="target", src_scope_name="main"):

    # main DQN과 target DQN의 Weight 호출
    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    op_holder = []
    for src_var, dest_var in zip(src_vars, dest_vars): # 하나씩 쌍을 맞춰서
        op_holder.append(dest_var.assign(src_var.value())) # 네트워크 업데이트 Assign 생성
    return op_holder    # Assign operation 리스트 반환

def bot_play(mainDQN):
    s = env.reset()
    step = 0
    while True:
        env.render()    # 환경 렌더링
        a = np.argmax(mainDQN.predict(s))   # Q 를 최대로 하는 행동 선택
        s, reward, done, _ = env.step(a)    # 렌더링된 환경에 적용하여 표시
        step += 1

        if done or step >= 1000:  # 카트가 도착하거나 1000번 이상 진행시 종료
            print("Total score: {}".format(step))
            break

env = gym.make('MountainCar-v0') # MountainCar 환경 생성
env._max_episode_steps = 1000    # 1 에피소드당 에피소드 최대 스탭 설정

dis = 0.99                  # 감가율 (Discounting factor)
e = 1                       # E-greedy 정책을 위한 Epsilon
e_decay = 0.995             # 상기 Epsilon 의 감소율
REPLAY_MEMORY = 50000       # 리플레이 메모리 공간
max_episodes = 2000         # 최대 에피소드 수
replay_buffer = deque()     # 관측 정보 저장

# 학습 진행 상황 시각화
episode_history = []
score_history = []

plt.ion()  # 실시간 그래프 출력을 위한 Interactive mode 실행
fig = plt.figure(1)  # 윈도우 팝업
sf = fig.add_axes([0.15, 0.1, 0.8, 0.8])  # 그래프 영역 설정
plt.title('Learning Progress', fontsize=16) # Title
plt.xlim([1, max_episodes])  # X Axis limit
plt.ylim([0, env._max_episode_steps])  # Y Axis limit

line, = sf.plot(episode_history, score_history, 'b-', lw=1) # 함수 line 설정
plt.ylabel('Steps')
plt.xlabel('Episode')

with tf.Session() as sess:
    input_size = env.observation_space.shape[0]  # DQN Input
    output_size = env.action_space.n  # DQN Output
    tf.set_random_seed(777)  # DQN 네트워크 초기화시 난수 생성 패턴 고정(Reproducibility)
    mainDQN = DQN(sess, input_size, output_size, name="main")  # Main DQN class 선언
    targetDQN = DQN(sess, input_size, output_size, name="target")  # Target DQN class 선언
    tf.global_variables_initializer().run()  # Main DQN, Target DQN의 weight 초기화

    copy_ops = network_update(dest_scope_name="target", src_scope_name="main") # DQN 동기화 연산 선언
    sess.run(copy_ops) # MainDQN과 Target DQN 동기화
    beginner = False # 시각화를 위한 변수1
    intermediate = False # 시각화를 위한 변수2

    for episode in range(max_episodes): # 에피소드 루프 시작
        e *= e_decay    # e-greedy의 epsilon 업데이트
        e = max(0.01, e) # epsilon의 최소값 0.01로 조정
        done = False
        step_count = 0      # step count 초기화
        state = env.reset() # 환경 초기화

        while not done:
            if np.random.rand(1) < e: # e-greedy에 의한 탐험 여부 결정
                action = env.action_space.sample() # 무작위 행동 실행
            else:
                action = np.argmax(mainDQN.predict(state)) # main DQN으로부터 예측된 Q를 최대로 하는 행동 선택
            next_state, reward, done, _ = env.step(action) # 행동 실행후 다음 상태, 보상, 결과 관측
            # Caution : 항상 Positive reward가 필요한것은 아님.
            # env.step 에서 나오는 reward가 있음.
            replay_buffer.append((state, action, reward, next_state, done)) # 경험 리플레이 저장
            if len(replay_buffer) > REPLAY_MEMORY: # 최대 리플레이 메모리 초과시
                replay_buffer.popleft() # 보관된 리플레이 순차적으로 삭제 후 새로운 리플레이 저장

            state = next_state  # 상태 정보 업데이트
            step_count += 1

        episode_history.append(episode + 1), score_history.append(step_count)  # 그래프 출력을 위한 에피소드 및 스탭수 저장

        line.set_xdata(episode_history), line.set_ydata(score_history)  # 그래프 데이터 할당
        plt.draw(), plt.pause(0.00001)  # 그래프 출력

        print("Episode: {} steps: {}, e: {}".format(episode, step_count, e))  # 에피소드 당 스탭수 및 epsilon 출력

        if np.mean(score_history[-min(10, len(score_history)):]) < 150:  # 최근 10회 에피소드의 평균 점수가 150 이하이면,
            bot_play(mainDQN)  # Mountain Car 애니메이션 실행 후
            break  # 프로그램 종료
        elif np.mean(score_history[-min(10, len(score_history)):]) <= 1000 and beginner == False: # 평균 점수가 900 이하이면,
            bot_play(mainDQN)
            beginner = True # 초심자!
        elif np.mean(score_history[-min(10, len(score_history)):]) <= 500 and intermediate == False: # 평균 점수가 900 이하이면,
            bot_play(mainDQN)
            intermediate = True # 초심자!

        if episode % 10 == 1:  # 매 10회 에피소드마다 Main Network 학습 진행
            for _ in range(50):  # 한번 학습 시 50 Epoch 실행
                minibatch = random.sample(replay_buffer, 10)  # 리플레이 메모리에서 10개의 랜덤 에피소드 추출 후,
                loss, _ = replay_train(mainDQN, targetDQN, minibatch)  # Main Network 학습

            print("Loss: ", loss)
            sess.run(copy_ops)  # Target DQN 업데이트