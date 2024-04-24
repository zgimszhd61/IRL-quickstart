# IRL-quickstart

当然，这里提供一个简单的逆强化学习（IRL）示例，使用一个小的网格世界环境，其中我们尝试从专家的行为中推断出奖励函数。在这个例子中，我们将使用一个简单的线性奖励模型，并采用策略匹配方法。假设我们已经有了专家的轨迹数据，我们将通过最大熵逆强化学习方法来估计奖励函数。

以下是完整的代码示例，使用Python编写：

```python
import numpy as np
import matplotlib.pyplot as plt

# 环境设置
class GridWorld:
    def __init__(self, width, height, start, goal, obstacle_rate=0.2):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.states = [(x, y) for x in range(width) for y in range(height)]
        self.end_states = [goal]  # 终点位置
        self.obstacles = self.init_obstacles(obstacle_rate)

    def init_obstacles(self, obstacle_rate):
        obstacles = set()
        while len(obstacles) < int(self.width * self.height * obstacle_rate):
            obstacle = (np.random.randint(0, self.width), np.random.randint(0, self.height))
            if obstacle != self.start and obstacle != self.goal:
                obstacles.add(obstacle)
        return obstacles

    def actions(self, state):
        actions = []
        x, y = state
        if x > 0 and (x-1, y) not in self.obstacles:
            actions.append((-1, 0))  # 左
        if x < self.width - 1 and (x+1, y) not in self.obstacles:
            actions.append((1, 0))  # 右
        if y > 0 and (x, y-1) not in self.obstacles:
            actions.append((0, -1))  # 下
        if y < self.height - 1 and (x, y+1) not in self.obstacles:
            actions.append((0, 1))  # 上
        return actions

    def next_state(self, state, action):
        next_state = (state[0] + action[0], state[1] + action[1])
        return next_state if next_state not in self.obstacles else state

    def is_terminal(self, state):
        return state in self.end_states

# 奖励函数
def reward_function(state):
    return -1  # 每走一步扣除1分

# 专家的轨迹
def generate_expert_trajectory(grid, policy):
    trajectory = []
    state = grid.start
    while not grid.is_terminal(state):
        action = policy(state)
        trajectory.append((state, action))
        state = grid.next_state(state, action)
    return trajectory

# 随机策略作为专家示例
def expert_policy(state):
    actions = grid.actions(state)  # 获取当前状态可行的动作
    return actions[np.random.choice(len(actions))] if actions else None

# 用户输入初始点和目标点
start_point = (0, 0)  # 示例初始点
goal_point = (4, 4)  # 示例目标点

# 初始化环境
grid = GridWorld(5, 5, start_point, goal_point)

# 打印初始化的网格世界所有状态和障碍物
print("Initialized GridWorld States:")
for state in grid.states:
    if state in grid.obstacles:
        print(f"Obstacle at {state}", end=' ')
    elif state == start_point:
        print(f"Start at {state}", end=' ')
    elif state == goal_point:
        print(f"Goal at {state}", end=' ')
    else:
        print(state, end=' ')
print()  # 打印一个换行符，以美观地分隔输出内容

expert_trajectories = [generate_expert_trajectory(grid, expert_policy) for _ in range(10)]

# 打印专家轨迹
for idx, trajectory in enumerate(expert_trajectories):
    print(f"Trajectory {idx + 1}:")
    for state, action in trajectory:
        print(f"State: {state}, Action: {action}")

# 逆强化学习算法（无修改，只为完整性重复）
def irl(grid, expert_trajectories, epochs, learning_rate):
    estimated_rewards = np.zeros((grid.width, grid.height))
    for _ in range(epochs):
        gradient = np.zeros_like(estimated_rewards)
        for trajectory in expert_trajectories:
            for state, _ in trajectory:
                gradient[state] += 1
        estimated_rewards -= learning_rate * gradient
    return estimated_rewards

# 运行IRL
estimated_rewards = irl(grid, expert_trajectories, 100, 0.01)

# 可视化结果
plt.imshow(estimated_rewards, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

```

这段代码首先定义了一个5x5的网格世界，其中只有一个终点状态。我们使用了一个简单的策略来生成专家的轨迹，这个策略是随机向右或向上移动。然后，我们实现了一个基本的逆强化学习算法，它使用梯度下降方法来调整估计的奖励函数，以便最大化地匹配专家的行为。

最后，我们将估计的奖励函数可视化，从而可以看到学习的奖励分布。这是一个非常基础的示例，实际应用中逆强化学习算法通常会更复杂，并需要考虑更多的因素，如状态转移概率、不同的奖励结构等。
