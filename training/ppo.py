"""
PPO-LSTM training loop — self-contained, no curriculum dependency.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import wandb

from .config import PPOConfig, TrainConfig, LoggingConfig
from .agent import Agent


# ---------------------------------------------------------------------------
# RL utilities
# ---------------------------------------------------------------------------

class AdaptiveScheduler:
    def __init__(self, kl_threshold: float = 0.008):
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr: float, kl_dist: float) -> float:
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma, reduce=True):
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)
    if reduce:
        return kl.mean()
    return kl


def bound_loss(mu, soft_bound: float = 1.1):
    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
    return b_loss.mean()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _build_checkpoint(agent, env, global_step, update, all_returns, all_raw_returns,
                      all_lengths, best_avg_return):
    ckpt = {
        "agent": agent.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "critic_normalizer": agent.critic_normalizer.state_dict(),
        "actor_learning_rate": agent.actor_optimizer.param_groups[0]["lr"],
        "critic_learning_rate": agent.critic_optimizer.param_groups[0]["lr"],
        "global_step": global_step,
        "update": update,
        "all_returns": all_returns,
        "all_raw_returns": all_raw_returns,
        "all_lengths": all_lengths,
        "best_avg_return": best_avg_return,
    }
    if hasattr(env, "normalizers"):
        normalizer_state = {}
        for k, norm in env.normalizers.items():
            normalizer_state[k] = {
                "mean": norm.mean,
                "var": norm.var,
                "count": norm.count,
                "clip_range": norm.clip_range,
            }
        ckpt["normalizer_state"] = normalizer_state
    return ckpt


def _restore_checkpoint(checkpoint_path, device, agent, env, ppo_cfg):
    if not os.path.exists(checkpoint_path):
        print(f"[WARNING] --resume: checkpoint not found at {checkpoint_path}. Starting from scratch.")
        return 0, 1, {}, -float("inf")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt["agent"])
    agent.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
    agent.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
    agent.critic_normalizer.load_state_dict(ckpt["critic_normalizer"])

    if hasattr(env, "normalizers") and "normalizer_state" in ckpt:
        for k, state in ckpt["normalizer_state"].items():
            if k in env.normalizers:
                env.normalizers[k].mean = state["mean"]
                env.normalizers[k].var = state["var"]
                env.normalizers[k].count = state["count"]
                env.normalizers[k].clip_range = state["clip_range"]

    agent.actor_optimizer.param_groups[0]["lr"] = ckpt.get("actor_learning_rate", ppo_cfg.learning_rate)
    agent.critic_optimizer.param_groups[0]["lr"] = ckpt.get("critic_learning_rate", ppo_cfg.learning_rate)

    global_step = ckpt.get("global_step", 0)
    start_update = ckpt.get("update", 1) + 1
    tracking = {
        "all_returns": ckpt.get("all_returns", []),
        "all_raw_returns": ckpt.get("all_raw_returns", []),
        "all_lengths": ckpt.get("all_lengths", []),
    }
    best_avg_return = ckpt.get("best_avg_return", -float("inf"))

    print(f"[INFO] Resumed from checkpoint at update {start_update - 1}, global_step {global_step}.")
    return global_step, start_update, tracking, best_avg_return


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    env,
    agent: Agent,
    train_cfg: TrainConfig,
    ppo_cfg: PPOConfig,
    log_cfg: LoggingConfig,
    resume: bool = False,
):
    device = torch.device(train_cfg.device)
    num_envs = train_cfg.num_envs
    num_steps = ppo_cfg.num_steps
    num_updates = ppo_cfg.num_updates
    batch_size = num_envs * num_steps

    assert num_envs % ppo_cfg.num_minibatches == 0, \
        f"num_envs ({num_envs}) must be divisible by num_minibatches ({ppo_cfg.num_minibatches})"

    # Dimensions
    policy_dim = env.total_obs_space["policy"].shape[-1]
    critic_dim = env.total_obs_space["critic"].shape[-1]
    action_dim = env.action_space.shape[-1]

    # --- Paths ---
    exp_checkpoint_dir = os.path.join(log_cfg.checkpoint_dir, train_cfg.exp_name)
    os.makedirs(exp_checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(exp_checkpoint_dir, "latest_model.pt")
    best_checkpoint_path = os.path.join(exp_checkpoint_dir, "best.pt")
    snapshot_interval = max(1, num_updates // log_cfg.snapshot_divisor)

    # --- WandB ---
    if log_cfg.wandb_enabled:
        init_kwargs = {
            "project": log_cfg.wandb_project,
            "name": f"{train_cfg.exp_name}_{int(time.time())}",
        }
        if log_cfg.wandb_group:
            init_kwargs["group"] = log_cfg.wandb_group
        wandb.init(**init_kwargs)

    # --- Rollout buffers ---
    obs = {
        "policy": torch.zeros((num_steps, num_envs, policy_dim), device=device),
        "critic": torch.zeros((num_steps, num_envs, critic_dim), device=device),
    }
    actions = torch.zeros((num_steps, num_envs, action_dim), device=device)
    log_probs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    terminateds = torch.zeros((num_steps, num_envs), device=device)
    truncateds = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)
    a_mus = torch.zeros((num_steps, num_envs, action_dim), device=device)
    a_stds = torch.zeros((num_steps, num_envs, action_dim), device=device)

    # --- Episode tracking ---
    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    raw_episode_returns = np.zeros(num_envs, dtype=np.float32)
    all_returns = []
    all_raw_returns = []
    all_lengths = []
    best_avg_return = -float("inf")
    global_step = 0
    start_update = 1

    # --- Resume ---
    if resume:
        global_step, start_update, tracking, best_avg_return = \
            _restore_checkpoint(checkpoint_path, device, agent, env, ppo_cfg)
        all_returns = tracking.get("all_returns", [])
        all_raw_returns = tracking.get("all_raw_returns", [])
        all_lengths = tracking.get("all_lengths", [])

    # --- LR scheduler ---
    adaptive_scheduler = AdaptiveScheduler(kl_threshold=ppo_cfg.target_kl) if ppo_cfg.adaptive_scheduler else None

    # --- LSTM states ---
    hidden_size = agent.hidden_size
    num_layers = agent.num_layers

    next_lstm_state_actor = (
        torch.zeros(num_layers, num_envs, hidden_size, device=device),
        torch.zeros(num_layers, num_envs, hidden_size, device=device),
    )
    next_lstm_state_critic = (
        torch.zeros(num_layers, num_envs, hidden_size, device=device),
        torch.zeros(num_layers, num_envs, hidden_size, device=device),
    )

    # --- Reset env ---
    env.train()
    next_obs, _ = env.reset()
    next_done = torch.zeros(num_envs, device=device)

    start_time = time.time()
    true_kl_dist = None

    # ============================
    # TRAINING LOOP
    # ============================
    for update in range(start_update, num_updates + 1):

        initial_lstm_state_actor = (next_lstm_state_actor[0].clone(), next_lstm_state_actor[1].clone())
        initial_lstm_state_critic = (next_lstm_state_critic[0].clone(), next_lstm_state_critic[1].clone())

        # ---- Rollout collection ----
        for step in range(num_steps):
            global_step += num_envs

            obs["policy"][step] = next_obs["policy"]
            obs["critic"][step] = next_obs["critic"]
            dones[step] = next_done

            with torch.no_grad():
                action, log_prob, _, next_lstm_state_actor, a_mu, a_std = \
                    agent.get_action(next_obs, next_lstm_state_actor, next_done)
                value, next_lstm_state_critic = \
                    agent.get_value(next_obs, next_lstm_state_critic, next_done, denorm=True, update_running_mean=False)
                values[step] = value.flatten()

            actions[step] = action
            log_probs[step] = log_prob
            a_mus[step] = a_mu
            a_stds[step] = a_std

            next_obs, reward, terminated, truncated, info_custom = env.step(action)
            next_done = (terminated | truncated).float()
            terminateds[step] = terminated.float()
            truncateds[step] = truncated.float()
            rewards[step] = reward

            # --- Episodic return tracking ---
            raw_reward = info_custom.get("org_reward", None)
            reward_np = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward
            done_np = (terminated | truncated).cpu().numpy() if isinstance(terminated, torch.Tensor) else (terminated | truncated)

            episode_returns += reward_np
            if raw_reward is not None:
                raw_episode_returns += raw_reward
            episode_lengths += 1

            if np.any(done_np):
                all_returns.extend(episode_returns[done_np == 1])
                all_raw_returns.extend(raw_episode_returns[done_np == 1])
                all_lengths.extend(episode_lengths[done_np == 1])

                episode_returns[done_np == 1] = 0
                raw_episode_returns[done_np == 1] = 0
                episode_lengths[done_np == 1] = 0

        # ---- GAE computation ----
        with torch.no_grad():
            next_value, _ = agent.get_value(
                next_obs, next_lstm_state_critic, next_done,
                denorm=True, update_running_mean=False,
            )
            next_value = next_value.reshape(1, -1)
            next_truncated = truncated.float()

            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                    is_truncated = next_truncated
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                    is_truncated = truncateds[t + 1]
                delta = rewards[t] + ppo_cfg.gamma * (nextnonterminal * nextvalues + is_truncated * values[t]) - values[t]
                advantages[t] = lastgaelam = delta + ppo_cfg.gamma * ppo_cfg.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # ---- Flatten batches ----
        b_obs = {
            "policy": obs["policy"].reshape(-1, policy_dim),
            "critic": obs["critic"].reshape(-1, critic_dim),
        }
        b_logprobs = log_probs.reshape(-1)
        b_a_mus = a_mus.reshape(-1, action_dim)
        b_a_stds = a_stds.reshape(-1, action_dim)
        b_actions = actions.reshape(-1, action_dim)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ---- PPO update ----
        envsperbatch = num_envs // ppo_cfg.num_minibatches
        envinds = np.arange(num_envs)
        flatinds = np.arange(batch_size).reshape(num_steps, num_envs)
        clipfracs = []

        for epoch in range(ppo_cfg.update_epochs):
            np.random.shuffle(envinds)

            for start in range(0, num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()
                mb_inds = torch.as_tensor(mb_inds, dtype=torch.long, device=device)

                mb_obs = {k: v[mb_inds] for k, v in b_obs.items()}
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_values = b_values[mb_inds]

                # Value normalization
                if ppo_cfg.norm_value:
                    agent.critic_normalizer.train()
                    mb_values = agent.critic_normalizer(mb_values, denorm=False)
                    mb_returns = agent.critic_normalizer(mb_returns, denorm=False)
                    agent.critic_normalizer.eval()

                # Forward pass
                _, new_logprob, entropy, _, a_mu, a_std = agent.get_action(
                    mb_obs,
                    (initial_lstm_state_actor[0][:, mbenvinds], initial_lstm_state_actor[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions[mb_inds],
                )

                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                newvalue, _ = agent.get_value(
                    mb_obs,
                    (initial_lstm_state_critic[0][:, mbenvinds], initial_lstm_state_critic[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    denorm=False,
                    update_running_mean=False,
                )

                # KL diagnostics
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > ppo_cfg.clip_coef).float().mean().item())

                with torch.no_grad():
                    true_kl_dist = policy_kl(b_a_mus[mb_inds], b_a_stds[mb_inds], a_mu, a_std)

                # Adaptive LR
                if ppo_cfg.anneal_lr and ppo_cfg.adaptive_scheduler:
                    if adaptive_scheduler is not None and true_kl_dist is not None:
                        lrnow = adaptive_scheduler.update(
                            agent.actor_optimizer.param_groups[0]["lr"], true_kl_dist
                        )
                        for pg in agent.actor_optimizer.param_groups:
                            pg["lr"] = lrnow
                elif ppo_cfg.anneal_lr:
                    frac = 1.0 - (update - 1.0) / num_updates
                    lrnow = frac * ppo_cfg.learning_rate
                    for pg in agent.actor_optimizer.param_groups:
                        pg["lr"] = lrnow

                # Advantage normalization
                if ppo_cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - ppo_cfg.clip_coef, 1 + ppo_cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if ppo_cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values, -ppo_cfg.clip_coef, ppo_cfg.clip_coef
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                boundary_loss = bound_loss(a_mu)

                loss = (
                    pg_loss
                    - ppo_cfg.ent_coef * entropy_loss
                    + v_loss * ppo_cfg.vf_coef
                    + boundary_loss * ppo_cfg.bound_loss_coef
                )

                agent.actor_optimizer.zero_grad()
                agent.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), ppo_cfg.max_grad_norm)
                agent.actor_optimizer.step()
                agent.critic_optimizer.step()

        # ---- Logging ----
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if log_cfg.wandb_enabled:
            wandb.log({
                "actor_learning_rate": agent.actor_optimizer.param_groups[0]["lr"],
                "critic_learning_rate": agent.critic_optimizer.param_groups[0]["lr"],
                "value_loss": v_loss.item(),
                "policy_loss": pg_loss.item(),
                "entropy_loss": entropy_loss.item(),
                "old_approx_kl": old_approx_kl.item(),
                "approx_kl": approx_kl.item(),
                "true_kl_dist": true_kl_dist.item() if true_kl_dist is not None else 0.0,
                "explained_variance": explained_var,
                "clipfrac": np.mean(clipfracs),
                "boundary_loss": boundary_loss.item(),
                "action_std/mean": b_a_stds.mean().item(),
                "action_std/min": b_a_stds.min().item(),
                "action_std/max": b_a_stds.max().item(),
                "action_mu/mean_abs": b_a_mus.abs().mean().item(),
                "action_mu/max_abs": b_a_mus.abs().max().item(),
                "mean_log_prob": b_logprobs.mean().item(),
            }, step=global_step)

        # ---- Episodic metrics ----
        if len(all_returns) > 0:
            avg_return = np.mean(all_returns[-100:])
            avg_raw_return = np.mean(all_raw_returns[-100:])
            avg_length = np.mean(all_lengths[-100:])

            metrics = {
                "avg_return": avg_return,
                "avg_length": avg_length,
                "avg_raw_return": avg_raw_return,
                "total_episodes_completed": len(all_returns),
            }

            if log_cfg.wandb_enabled:
                wandb.log(metrics, step=global_step)

            print(
                f"Update {update}/{num_updates}, Step: {global_step}, "
                f"Return: {avg_return:.2f}, Raw: {avg_raw_return:.2f}, Length: {avg_length:.1f}, "
                f"Time: {time.time() - start_time:.2f}s"
            )

            # ---- Best model saving ----
            if avg_return > best_avg_return:
                best_avg_return = avg_return
                best_ckpt = _build_checkpoint(
                    agent, env, global_step, update,
                    all_returns, all_raw_returns, all_lengths,
                    best_avg_return,
                )
                torch.save(best_ckpt, best_checkpoint_path)
                print(f"[INFO] Best model saved: return={best_avg_return:.2f}")

        # ---- Latest model checkpoint ----
        if update % log_cfg.checkpoint_interval == 0:
            ckpt = _build_checkpoint(
                agent, env, global_step, update,
                all_returns, all_raw_returns, all_lengths,
                best_avg_return,
            )
            torch.save(ckpt, checkpoint_path)

        # ---- Snapshot checkpoint ----
        if update % snapshot_interval == 0:
            snapshot_path = os.path.join(exp_checkpoint_dir, f"{global_step}.pt")
            snap_ckpt = _build_checkpoint(
                agent, env, global_step, update,
                all_returns, all_raw_returns, all_lengths,
                best_avg_return,
            )
            torch.save(snap_ckpt, snapshot_path)
            print(f"[INFO] Snapshot saved: {snapshot_path}")

    # ---- Cleanup ----
    if log_cfg.wandb_enabled:
        wandb.finish()
