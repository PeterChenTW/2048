
# 2048 Reinforcement Learning Project

This project consists of a 2048 game implementation and a program that trains a game AI using reinforcement learning.

## Contents

- `main.py`: The main entry point of the application. This file contains the code to start the game and train the AI.
- `game_2048.py`: The implementation of the 2048 game, including the game logic and the UI interface.
- `RL.py`: The implementation of the reinforcement learning. This file contains the code to train the AI.

## How to Run the Game

Run `main.py` to start the game. The goal of the game is to slide the number blocks on a 4x4 grid. Each slide randomly generates a 2 or 4 in an empty position. When two blocks with the same number collide, they merge into their sum. The game ends when the board is filled with numbers and no valid moves can be made.

```bash
python main.py
```
## Flow
```mermaid
graph TD
    A[Initialize agent, env, records]
    B[Train for episodes]
    C[Get initial state]
    D[Agent chooses action]
    E[Env executes action, returns reward etc.]
    F[Store transition] 
    G[Update state]
    H{Done?}
    I[Record and print episode info]
    J[Agent replays memory]
    K[Update target net periodically]
    L[Reset env]
    M[Save model periodically]
    
    A-->B
    B-->C
    C-->D
    D-->E
    E-->F
    E-->G
    G-->H
    H--Yes-->I
    I-->L 
    L-->B
    H--No-->D
    
    B-->J
    J-->K
    K-->M
```
## Results

Here are the results of the training process:

![Results](result/nn_prioritize_replay_no_invalid_gamma_099_2.png)

This plot contains four subplots showing the game score, loss, game turns, and game invalid turn ratio, respectively, as the training progresses.

## How to Train the AI

If you wish to train your own AI, you can run the `train_DDQN` function in `main.py`. This will train the AI using reinforcement learning.

```bash
python main.py
```

Note that the training process might take some time and will vary depending on the performance of your system.

## Lessons Learned

While developing this project, a few key lessons were learnt:

- Removing invalid moves helped in reducing the action space, thereby making the AI training process more efficient.
- An excessively high initial learning rate and a wrong optimizer can easily lead to hitting the boundary. It's crucial to monitor the loss during the training process.
- The size of epsilon in the reinforcement learning process is an important parameter.
- By fixing the random seed at the beginning, the training process can be made deterministic and reproducible.

---

Here is the AI in action:

![AI Playing 2048](game_recording.gif)

---

