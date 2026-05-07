﻿# HW3 Execution Architecture (DQN and Variants)

## 1) Scope

This file breaks HW3 into an execution-ready structure:

- HW3-1: Naive DQN for `static` mode (30%)
- HW3-2: Enhanced DQN variants for `player` mode (40%)
- HW3-3: Enhanced DQN for `random` mode with training tips (30%)
- HW3-4: Bonus - Rainbow DQN for `random` mode

## 2) Current Baseline in Repo

Existing files:

- `base/Ch3_book.ipynb`
- `base/GridBoard.py`
- `base/Gridworld.py`

Use `base/Ch3_book.ipynb` as the starting baseline and refactor into reusable scripts/modules.

## 3) Directory Layout

```text
G1-DRL-HW3-DQNAndItsVariants/
|- base/
|  |- Ch3_book.ipynb
|  |- GridBoard.py
|  `- Gridworld.py
|- src/
|  |- common/
|  |  |- env_adapter.py
|  |  |- replay_buffer.py
|  |  |- networks.py
|  |  |- trainer_utils.py
|  |  `- eval.py
|  |- hw3_1_naive_dqn_static.py
|  |- hw3_2_double_dqn_player.py
|  |- hw3_2_dueling_dqn_player.py
|  |- hw3_3_port_random.py
|  `- hw3_4_rainbow_random_bonus.py
|- configs/
|  |- hw3_1_static.yaml
|  |- hw3_2_player_double.yaml
|  |- hw3_2_player_dueling.yaml
|  |- hw3_3_random_port.yaml
|  `- hw3_4_rainbow_bonus.yaml
|- runs/
|  |- hw3_1/
|  |- hw3_2/
|  |- hw3_3/
|  `- hw3_4_bonus/
|- reports/
|  |- hw3_1_understanding.md
|  |- hw3_2_comparison.md
|  |- hw3_3_conversion_and_tips.md
|  `- hw3_4_rainbow_bonus.md
`- HW3_EXECUTION_ARCHITECTURE.md
```

## 4) Execution Roadmap

### Phase A: HW3-1 Naive DQN (`static`)

Goals:

- Run baseline DQN successfully
- Verify Experience Replay Buffer flow
- Submit a short understanding report

Outputs:

- Training script or notebook section for HW3-1
- `reports/hw3_1_understanding.md`
- Training logs/plots under `runs/hw3_1/`

Acceptance:

- Agent can solve `static` mode consistently
- Report explains state, action, reward, Q-target, and replay update flow

### Phase B: HW3-2 DQN Variants (`player`)

Goals:

- Implement `Double DQN`
- Implement `Dueling DQN`
- Compare both against Naive DQN

Outputs:

- `src/hw3_2_double_dqn_player.py`
- `src/hw3_2_dueling_dqn_player.py`
- `reports/hw3_2_comparison.md`

Acceptance:

- Double DQN target logic is correct (online selects action, target evaluates value)
- Dueling architecture is correct (Value + Advantage merge)
- Comparison clearly explains improvements over Naive DQN

### Phase C: HW3-3 Conversion + Training Tips (`random`)

Goals:

- Convert DQN implementation to either `Keras` or `PyTorch Lightning`
- Add at least one or two stabilization techniques

Techniques used in this phase:

- Gradient clipping
- Learning rate scheduler
- Improved epsilon decay strategy
- Better target network sync frequency

Outputs:

- `src/hw3_3_port_random.py`
- `reports/hw3_3_conversion_and_tips.md`
- Training logs/plots under `runs/hw3_3/`

Acceptance:

- Converted version can train and evaluate end-to-end
- Added techniques are active and discussed with evidence

### Phase D: HW3-4 Bonus Rainbow DQN (`random`)

Goal:

- Implement Rainbow DQN (or a practical subset) for random mode

MVP subset:

- Double DQN
- Dueling Network
- Prioritized Replay

Outputs:

- `src/hw3_4_rainbow_random_bonus.py`
- `reports/hw3_4_rainbow_bonus.md`

Acceptance:

- Report clearly states which Rainbow components were implemented
- Includes comparison vs HW3-2/HW3-3 models

## 5) Unified Experiment Protocol

To keep comparisons fair:

- Use fixed random seeds (at least 3 seeds)
- Use the same max episodes for compared runs
- Evaluate every N episodes with the same schedule
- Track: `episode_reward`, `win_rate`, `loss`, `epsilon`

For each task, keep at least:

- One reward curve figure
- One final test summary

## 6) Deliverable Checklist

`HW3-1`

- [x] Naive DQN runs in `static` mode
- [x] Experience Replay is used
- [x] Short understanding report is complete

`HW3-2`

- [x] Double DQN is implemented
- [x] Dueling DQN is implemented
- [x] Comparison with Naive DQN is complete

`HW3-3`

- [x] Converted to Keras or PyTorch Lightning
- [x] At least one stabilization tip is integrated
- [x] Results and analysis are documented

`HW3-4 Bonus`

- [x] Rainbow DQN (or subset) runs in `random` mode
- [x] Bonus comparison and discussion are documented


