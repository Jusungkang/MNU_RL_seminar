##
## 작성일 : 20231123
## 작성자 : Jusung Kang
## 목적 : CartPole 예제 셈플 코드 주석 작성
## 참조 URL : N/A
##

## open AI Gym 포함 필요 library 호출
import gymnasium as gym
import numpy as np
from time import sleep

## CartPole Envrionment 호출 함수
env = gym.make("CartPole-v1", render_mode="human")

## Env. 초기화
env.reset()

for _ in range(50):

    ## Action Space 함수
    # action = env.step(0)   ## left side
    # action = env.step(1)   ## right side
    action = env.action_space.sample()

    ## Env. 로부터 전달받는 Observation 결과
    Observation = env.step(action)
    print('Current Observation is : ' + str(Observation[0]))

    ## GUI를 통한 결과 시각화
    env.render()

    ## Terminate state 발생 시 Env. 를 초기화
    if np.abs(Observation[2]) == True:
        env.reset()

    ## 부드러운 시각화를 위한 Time sleep
    sleep(0.03)

## Env 종료
env.close()
