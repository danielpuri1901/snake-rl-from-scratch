# Snake RL from scratch

A small neural network learns to play a simple Snake game with reinforcement learning.
The Python version uses only NumPy, and the C version uses no libraries.

Both versions use REINFORCE.
The agent plays 64 games, scores each move by the reward that came after it, and makes good moves more likely.

The snake has no tail and moves on a 6x6 grid.
It gets +20 for food and -10 for leaving the grid, and each game stops at 100 steps.

## Run

Python, from the repo root:

```sh
python3 autograd.py            # check the gradients
python3 env.py --epochs 10000  # train
```

C:

```sh
cd c
clang env.c -o env -lm
./env
```

## Results

Python, 10,000 epochs, seed 0. Each value is the average over 100 epochs.

| | First 100 epochs | Last 100 epochs |
|---|---|---|
| Foods per game | 0.2 | 21.7 |
| Steps per game | 10.7 | 98.0 |
| Games that end in a crash | 100% | 4.4% |

## Credit

`c/` is an unmodified copy of [harshbhatt7585/cRL](https://github.com/harshbhatt7585/cRL) at commit `f8fac26`.
That repo has no license, so this repo does not grant one for `c/`.

The Python version is a port of that code, written while following [this freeCodeCamp course](https://www.youtube.com/watch?v=R5fPn6W8m6k).
