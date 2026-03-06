import random
import time

BOARD_SIZE = 4

# =========================================================
# GLOBAL AGENT REGISTRY (runtime only)
# =========================================================
AGENTS = {}  # key: initial puzzle (tuple), value: QAgent


# -------------------------
# Puzzle Environment
# -------------------------
class PuzzleEnv:
    def __init__(self, initial_state, ordem="standard"):
        self.initial = [row[:] for row in initial_state]
        self.state = [row[:] for row in initial_state]

        if ordem == "zero_first":
            self.goal = [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15]
            ]
        else:  # standard
            self.goal = [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 0]
            ]
        self.moves = 0

    def reset(self):
        self.state = [row[:] for row in self.initial]
        self.moves = 0
        return self.get_state()

    def get_zero(self):
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.state[i][j] == 0:
                    return i, j

    def get_actions(self):
        i, j = self.get_zero()
        actions = []
        if i > 0: actions.append(0)  # up
        if i < BOARD_SIZE - 1: actions.append(1)  # down
        if j > 0: actions.append(2)  # left
        if j < BOARD_SIZE - 1: actions.append(3)  # right
        return actions

    def step(self, action):
        i, j = self.get_zero()
        ni, nj = i, j

        if action == 0 and i > 0: ni -= 1
        elif action == 1 and i < BOARD_SIZE - 1: ni += 1
        elif action == 2 and j > 0: nj -= 1
        elif action == 3 and j < BOARD_SIZE - 1: nj += 1
        else:
            return self.get_state(), -10, False

        self.state[i][j], self.state[ni][nj] = self.state[ni][nj], self.state[i][j]
        self.moves += 1

        done = self.state == self.goal
        reward = self.calc_reward(done)
        return self.get_state(), reward, done

    def calc_reward(self, done):
        if done:
            return 1000

        # Manhattan distance
        dist = 0
        for i, row in enumerate(self.state):
            for j, val in enumerate(row):
                if val != 0:
                    gi = (val - 1) // 4
                    gj = (val - 1) % 4
                    dist += abs(gi - i) + abs(gj - j)
        return -dist

    def get_state(self):
        return tuple(val for row in self.state for val in row)


# -------------------------
# Q-Learning Agent
# -------------------------
class QAgent:
    def __init__(self, alpha=0.5, gamma=0.95, epsilon=1.0):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 4
        return self.q_table[state]

    def choose_action(self, state, possible_actions):
        if random.random() < self.epsilon:
            return random.choice(possible_actions)

        q_values = self.get_q(state)
        best = max(q_values[a] for a in possible_actions)
        best_actions = [a for a in possible_actions if q_values[a] == best]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_actions):
        q = self.get_q(state)
        max_next = max(self.get_q(next_state)[a] for a in next_actions) if next_actions else 0
        q[action] += self.alpha * (reward + self.gamma * max_next - q[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# -------------------------
# Training Phase
# -------------------------
def train_agent(matrix, agent, episodes=10000, max_steps=200, ordem="standard"):
    env = PuzzleEnv(matrix, ordem)
    expansions = 0
    start_time = time.time()

    for _ in range(episodes):
        state = env.reset()
        last_state = None

        for _ in range(max_steps):
            expansions += 1
            actions = env.get_actions()
            action = agent.choose_action(state, actions)

            next_state, reward, done = env.step(action)
            next_actions = env.get_actions()

            if next_state == last_state:
                actions.remove(action)
                if actions:
                    action = random.choice(actions)
                    next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state, next_actions)
            state, last_state = next_state, state

            if done:
                break

        agent.decay_epsilon()

    total_time = time.time() - start_time
    print(f"Training done. Expansions: {expansions}, Time: {total_time:.2f}s")
    return agent, expansions, total_time


# -------------------------
# Solving Phase
# -------------------------
def solve_with_agent(matrix, agent, max_steps=500, ordem="standard"):
    env = PuzzleEnv(matrix, ordem)
    state = env.reset()
    path = [[row[:] for row in env.state]]
    last_state = None
    expansions = 0
    start_time = time.time()

    for _ in range(max_steps):
        expansions += 1
        actions = env.get_actions()
        action = agent.choose_action(state, actions)

        next_state, _, done = env.step(action)

        if next_state == last_state:
            actions.remove(action)
            if actions:
                action = random.choice(actions)
                next_state, _, done = env.step(action)

        path.append([row[:] for row in env.state])

        if done:
            total_time = time.time() - start_time
            print(f"Solução encontrada!")
            print(f"Time to solve: {total_time:.6f}s")
            print(f"Solution moves: {len(path)-1}")
            print(f"Expansions during solve: {expansions}")
            return path

        last_state = state
        state = next_state

    total_time = time.time() - start_time
    print(f"RL agent failed to solve within max_steps")
    print(f"Time during solve: {total_time:.2f}s")
    print(f"Expansions during solve: {expansions}")
    return None


# -------------------------
# Main Resolution Function
# -------------------------
def resolucao_RL(matrix, ordem="standard", episodes=10000, max_steps=200):
    puzzle_key = tuple(val for row in matrix for val in row)

    if puzzle_key not in AGENTS:
        AGENTS[puzzle_key] = QAgent()

    agent = AGENTS[puzzle_key]

    agent, train_exp, train_time = train_agent(matrix, agent, episodes, max_steps, ordem)
    solution = solve_with_agent(matrix, agent, max_steps, ordem)

    if solution is None:
        print("Call resolucao_RL again to continue training.")

    return solution