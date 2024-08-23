import numpy as np
import gym

# 환경 설정 (OpenAI Gym의 FrozenLake 환경)
env = gym.make("FrozenLake-v1", is_slippery=False)

# Q-테이블 초기화
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# 학습 파라미터
alpha = 0.8    # 학습률
gamma = 0.95   # 할인률
epsilon = 0.1  # 탐험(exploration) 확률
episodes = 1000

# Q-learning 알고리즘
for i in range(episodes):
    state = env.reset()
    # 만약 state가 튜플이면 첫 번째 값을 사용
    if isinstance(state, tuple):
        state = state[0]
    done = False
    
    while not done:
        # 상태 값의 유효성 확인
        if not (0 <= state < q_table.shape[0]):
            print(f"Invalid state: {state}")
            break
        
        # ε-greedy 정책에 따라 행동 선택
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # 무작위 행동 선택
        else:
            action = np.argmax(q_table[state])  # Q-값이 가장 큰 행동 선택
        
        # 환경과 상호작용
        next_state, reward, terminated, truncated, _ = env.step(action)
        # 만약 next_state가 튜플이면 첫 번째 값을 사용
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        
        # Q-테이블 업데이트
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
        
        # 상태 업데이트
        state = next_state
        done = terminated or truncated  # 에피소드 종료 조건

print("학습된 Q-테이블:")
print(q_table)

