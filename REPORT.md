# Inverted Pendulum RL — Report

## Algorithm

Both tasks use a custom **PPO (Proximal Policy Optimization)** implementation with an **LSTM-based Actor-Critic** architecture. The actor and critic are separate networks, each consisting of a single-layer LSTM (hidden size 64) followed by MLP heads (64 -> 32). The policy outputs a continuous 1D action (force on the cart) with a learned log-std for exploration.

Key training details:
- **Observation encoding:** cos/sin of joint angles to avoid wrapping discontinuities, plus joint velocities and cart state
- **Reward:** `cos(angle)` per arm for upright incentive, with small penalties on cart position, velocities, and a termination penalty for cart out-of-bounds
- **No pole angle termination:** the agent is free to swing through any angle, only terminated when the cart exceeds the rail bounds
- **Observation normalization** via running mean/variance (no reward normalization)
- **Adaptive LR scheduling** with target KL divergence (0.008)
- **256 parallel environments**, rollout length of 256 steps, 5 update epochs per rollout

## Task 1: Single Inverted Pendulum Swing-Up

The pole starts hanging downward (angle = pi) with random perturbations and the agent must swing it up and balance it at the upright position (angle = 0).

- **Observation space (5D):** cos(pole), sin(pole), pole_vel, cart_pos, cart_vel
- **Action:** 1D force on cart, scaled to 130N
- **Episode length:** 15 seconds
- **Network:** LSTM (1 layer, hidden 64) + Actor MLP [64, 32] + Critic MLP [64, 32]
- **Training:** 256 envs, rollout length 256, entropy coeff 0.01

### Results

The agent learned to swing up and balance from any starting position within approximately **3M environment steps**. It reliably swings the pole up using 1-2 swings and stabilizes it at the upright position.

**Training curves:** [WandB — Task 1](https://api.wandb.ai/links/aadithya-indian-institute-of-technology-jodhpur/ubdklbra)

**Video of trained agent:** [Task 1 Video](<https://drive.google.com/drive/folders/1vEy-0oNcEE5IXS1z4yfZK8Jj2A9HZGm7?usp=drive_link>)

## Task 2: Double Inverted Pendulum Swing-Up

Both arms start hanging downward and the agent must swing them up and balance both in the upright position using only the cart force.

- **Observation space (8D):** cos(arm1), sin(arm1), arm1_vel, cos(arm2), sin(arm2), arm2_vel, cart_pos, cart_vel
- **Action:** 1D force on cart, scaled to 130N
- **Episode length:** 15 seconds
- **Network:** LSTM (1 layer, hidden 64) + Actor MLP [64, 32] + Critic MLP [64, 32] (same as Task 1)
- **Training:** 256 envs, rollout length 256, entropy coeff 0.01
- **URDF asset:** custom double pendulum based on [DoublePendulumIsaacLab](https://github.com/NRdrgz/DoublePendulumIsaacLab) with modified joint origins so 0 rad = upright

### Results

The agent achieved high rewards at around **3M steps**, but exhibited a suboptimal strategy (e.g. holding one arm up while the other oscillated). As training continued to **6M steps**, the agent converged to a more coordinated swing-up and balance behavior for both arms.

**Training curves:** [WandB — Task 2](https://api.wandb.ai/links/aadithya-indian-institute-of-technology-jodhpur/ubdklbra)

**Video of trained agent:** [Task 2 Video](<https://drive.google.com/drive/folders/1HqQZw33MYA4W610vsfsB9ICbO-jGGOf3?usp=drive_link>)

## Limitations

The current system is not physically realizable:

- **No damping on passive joints:** The revolute joints for both arms have zero damping, meaning there is no friction or resistance to rotation. A real system would have some joint friction.
- **High cart force:** The agent was trained with a 130N action scale. Lower, more realistic force limits were not explored.
- **No action smoothing:** The policy can output rapid force changes between timesteps, which would be infeasible for a real actuator.
- **Idealized physics:** No sensor noise, no actuator delays, no model mismatch.

## Future Work

Given more time, the following experiments would be worth exploring:

- **Smaller network / no LSTM:** Test whether a simpler feedforward MLP (without LSTM) can solve these tasks, since the observation already contains velocities and the system is (mostly) Markovian. This would reduce inference latency for real-time deployment.
- **Physically realizable system:**
  - Add damping to the passive revolute joints to simulate real joint friction
  - Reduce the cart force limit (e.g. 50N or lower) and retrain
  - Add action rate penalties or action smoothing (e.g. penalize `|a_t - a_{t-1}|`) to produce smoother control signals
  - These changes would likely require restructuring the reward to produce smoother control
  - Add domain randomization (mass, friction, actuator delays) to make the policy robust enough for a real hardware setup
- **Energy-aware reward shaping:** The current `cos(angle)` reward encourages high potential energy (upright) but doesn't penalize excess kinetic energy near the top, leading to overshoot and oscillation. A complementary reward term like `exp(-k * |vel * angle|)` would address this — it allows high velocity when the pole is low (during the pumping phase) but penalizes velocity near the upright position (during the catching phase). This naturally captures the two-phase swing-up strategy without explicitly computing system energy. A naive total-energy reward like `exp(-k * |E - E_target|)` has a degeneracy problem: the agent can satisfy `KE + PE = PE_max` with the wrong energy distribution (e.g. cart moving fast, pole halfway up).

