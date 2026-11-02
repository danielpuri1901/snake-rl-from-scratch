"""Snake game and REINFORCE training loop (cRL: env.c)."""
import argparse
from autograd import backward_pass, forward_pass
from model import ActorModel
import numpy as np

LEFT, RIGHT, UP, DOWN, NONE = range(5)
MOVES = {LEFT: (-1, 0), RIGHT: (1, 0), UP: (0, 1), DOWN: (0, -1)}


class SnakeENV:
    def __init__(self, grid_size, rng):
        side = int(round(np.sqrt(grid_size)))
        if side * side != grid_size:
            raise ValueError(f"grid_size must be a perfect square, got {grid_size}")
        self.rows = self.cols = side
        self.grid_size = grid_size
        self.rng = rng
        self.reset_state()

    def random_cell(self):
        return (int(self.rng.integers(self.cols)), int(self.rng.integers(self.rows)))

    def reset_state(self):
        self.snake = (self.cols // 2, self.rows // 2)
        self.food = self.random_cell()
        while self.food == self.snake:
            self.food = self.random_cell()
        self.score = 0.0
        self.foods_eaten = 0
        self.pov = RIGHT

    def take_action(self, action):
        if action == NONE:
            action = self.pov
        dx, dy = MOVES[action]
        self.snake = (self.snake[0] + dx, self.snake[1] + dy)
        self.pov = action

    def game_over(self):
        x, y = self.snake
        return not (0 <= x < self.cols and 0 <= y < self.rows)

    def get_reward(self):
        reward = 0.0
        if self.snake == self.food:
            reward += 20.0
            self.foods_eaten += 1
            while self.food == self.snake:
                self.food = self.random_cell()
        if self.game_over():
            reward -= 10.0
        return reward

    def state_vector(self):
        """76 numbers: snake cell (36) + food cell (36) + direction (4) (cRL: build_state_vector)."""
        v = np.zeros((2 * self.grid_size + 4, 1), np.float32)
        x, y = self.snake
        fx, fy = self.food
        v[y * self.cols + x] = 1.0
        v[self.grid_size + fy * self.cols + fx] = 1.0
        v[2 * self.grid_size + self.pov] = 1.0
        return v

def sample_action(probs, rng):
    """Pick an action at random, weighted by the network's probabilities (cRL: sample_action)."""
    i = int(np.searchsorted(np.cumsum(probs), rng.random()))
    return min(i, len(probs) - 1)

def play(model, envs, rng, episode_len=100):
    """Play one game per env with the current network. Returns, per game, a list of (state, action, reward)."""
    # cRL plays the games one after another. Here they step together, so each
    # time step is one forward pass on a (76, games-still-alive) matrix. Same math.
    trajectories = [[] for _ in envs]
    for env in envs:
        env.reset_state()
    alive = list(range(len(envs)))

    for t in range(episode_len):
        if not alive:
            break
        states = [envs[i].state_vector() for i in alive]
        model.input.val = np.hstack(states)
        forward_pass(model.forward_graph)

        still_alive = []
        for col, i in enumerate(alive):
            env = envs[i]
            action = sample_action(model.output.val[:, col], rng)
            env.take_action(action)
            reward = env.get_reward()
            env.score += reward
            trajectories[i].append((states[col], action, reward))
            if not env.game_over():
                still_alive.append(i)
        alive = still_alive
    return trajectories

def score(trajectories, gamma=0.99):
    """Return G for every step, then normalize into an advantage."""
    states, actions, returns = [], [], []
    average_return = 0.0
    for traj in trajectories:
        G = 0.0
        for state, action, reward in reversed(traj):
            G = reward + gamma * G
            states.append(state)
            actions.append(action)
            returns.append(G)
        average_return += G   # after the loop, G is the return from the first step
    returns = np.array(returns, np.float32)
    return_std = returns.std()
    advantages = (returns - returns.mean()) / (return_std + 1e-8)
    return states, actions, advantages, average_return / len(trajectories), return_std

def learn(model, states, actions, advantages, learning_rate=0.05):
    """One forward + backward over every saved step at once, then one update of every weight."""
    # Each column is one step. matmul adds up the gradients over the columns,
    # the same total cRL gets by looping over the steps one by one.
    n = len(advantages)
    model.input.val = np.hstack(states)
    advantage = np.zeros((5, n), np.float32)
    advantage[actions, np.arange(n)] = advantages   # only the taken action's slot
    model.advantage.val = advantage

    for p in model.parameters:
        p.grad[:] = 0
    forward_pass(model.cost_graph)
    backward_pass(model.cost_graph)

    for p in model.parameters:
        p.val -= (learning_rate / n) * p.grad

def train(model, rng, epochs=2000, rollout_size=64, grid_size=36):
    envs = [SnakeENV(grid_size, rng) for _ in range(rollout_size)]
    for epoch in range(epochs):
        trajectories = play(model, envs, rng)
        states, actions, advantages, average_return, return_std = score(trajectories)
        learn(model, states, actions, advantages)

        foods = sum(env.foods_eaten for env in envs)
        print(f"Epoch {epoch} | Average return: {average_return:.3f} | "
              f"Foods: {foods} | Samples: {len(advantages)} | Return std: {return_std:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    train(ActorModel(rng), rng, epochs=args.epochs)