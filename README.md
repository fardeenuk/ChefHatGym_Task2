# ChefHatGym_Task2
# Task 2 — Reinforcement Learning in Chef’s Hat Gym (Variant: Student ID mod 7 = 0)

This repository contains my **Task 2** implementation for the **Chef’s Hat Gym** multi-agent card game environment.  
My **Student ID remainder = 0**, so the focus of this task is **opponent behaviour / opponent diversity**: training and evaluating an RL agent against different opponent types and reporting comparative results.

---

## 1) Variant (Student ID % 7)
- **Remainder:** 0  
- **Variant focus:** Opponent behaviour comparison (Random vs Mixed opponents)

---

## 2) Environment Summary
Chef’s Hat Gym is a **multi-agent** environment (4 players). The agent receives an observation containing game state information and an **action mask** representing legal moves, and must output a discrete action (encoded as a one-hot vector by the environment interface).

**Key challenge:** Multi-agent non-stationarity and a large discrete action space make training unstable for naïve RL methods.  

---

## 3) Opponent Configurations (Variant Requirement)
I evaluate the same agent under two opponent settings:

### A) Random Opponents
- 3 × built-in random agents (`AgentRandon`)

### B) Mixed Opponents
- 1 × random agent  
- 1 × **Greedy** heuristic opponent (selects highest-index valid action using action mask)  
- 1 × **Conservative** heuristic opponent (selects lowest-index valid action using action mask)

This directly supports the remainder=0 requirement (comparing performance against different opponent behaviours).

---

## 4) RL Method
### Algorithm
- **REINFORCE (policy gradient)** with:
  - **Action masking** (invalid actions suppressed before sampling)
  - **Moving-average baseline** to reduce gradient variance and improve stability

### Why the baseline was added
Vanilla REINFORCE can degrade performance in multi-agent settings due to high-variance gradients.  
Using a moving baseline improves stability and produced better results in this project.

---

## 5) Metric
The environment returns:
- `Game_Performance_Score`: a list of 4 floats (one per player)

Since the RL agent is always placed as the **first player**, the agent’s score is taken as:
- `Game_Performance_Score[0]`

Evaluation reports **mean ± std** over multiple games.

---

## 6) Results (mean ± std)
| Stage | Opponents | Mean | Std |
|------|-----------|------:|----:|
| Baseline | Random | 0.688 | 0.277 |
| Baseline | Mixed | 0.910 | 0.215 |
| Trained (REINFORCE + baseline) | Random | 0.718 | 0.298 |
| Trained (REINFORCE + baseline) | Mixed | 0.906 | 0.242 |

**Summary:** Training improved performance vs **Random** opponents and maintained similar performance vs **Mixed** opponents.

---

## 7) Training Curve
The training curve (train vs Random opponents) is saved at:

- `task2_outputs/training_curve_random.png`

Add this image to the README by committing the file and using:

``markdown
![Training Curve](task2_outputs/training_curve_random.png)

How to Run
A) Requirements

Recommended: Python 3.10 (or 3.11)

Install dependencies:

pip install -U pip setuptools wheel
pip install numpy==1.26.1 gym==0.26.2 gym-notices
pip install chefshatgym
pip install torch matplotlib pandas
B) Run Training + Evaluation

This project was run using a Jupyter Notebook / script. The workflow is:

Baseline evaluation vs Random and Mixed opponents

Train agent vs Random opponents (REINFORCE + baseline)

Evaluate trained agent vs Random and Mixed opponents

Save outputs to task2_outputs/
