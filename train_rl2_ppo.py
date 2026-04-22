import argparse
import csv
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl2_wrapper import RL2MetaWrapper


@dataclass
class PPOConfig:
    total_timesteps: int = 200_000
    rollout_steps: int = 256
    update_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 128
    seed: int = 0
    device: str = "cpu"
    results_dir: str = "results"
    exp_name: str = "rl2_ppo"
    save_every_updates: int = 25
    eval_every_updates: int = 10
    eval_trials: int = 20

    # env / task settings
    size: int = 5
    max_steps: int = 75
    step_penalty: float = 0.0
    success_reward: float = 1.0
    wrong_terminal_reward: float = 0.0
    object_types: int = 4
    object_colors: int = 4
    max_depth: int = 2
    prune_prob: float = 0.1
    num_distractor_rules: int = 0
    num_distractor_objects: int = 0
    episodes_per_trial: int = 4


class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=False)
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_state(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, 1, self.hidden_size, device=device)
        c = torch.zeros(1, 1, self.hidden_size, device=device)
        return h, c

    def forward_step(
        self,
        obs: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.encoder(obs)
        x = x.unsqueeze(0)  # (1, batch=1, hidden)
        out, next_state = self.lstm(x, state)
        out = out.squeeze(0)
        logits = self.policy_head(out)
        value = self.value_head(out).squeeze(-1)
        return logits, value, next_state

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        action: Optional[torch.Tensor] = None,
    ):
        logits, value, next_state = self.forward_step(obs, state)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy, value, next_state

    def evaluate_sequence(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        dones: torch.Tensor,
        initial_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a rollout as one recurrent sequence for PPO updates."""
        state = initial_state
        logprobs = []
        entropies = []
        values = []

        for t in range(obs.shape[0]):
            logits, value, next_state = self.forward_step(obs[t].unsqueeze(0), state)
            dist = Categorical(logits=logits)
            logprobs.append(dist.log_prob(actions[t].unsqueeze(0)).squeeze(0))
            entropies.append(dist.entropy().squeeze(0))
            values.append(value.squeeze(0))

            if dones[t].item() > 0.5:
                state = self.initial_state(obs.device)
            else:
                state = next_state

        return torch.stack(logprobs), torch.stack(entropies), torch.stack(values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_env(cfg: PPOConfig) -> RL2MetaWrapper:
    return RL2MetaWrapper(
        episodes_per_trial=cfg.episodes_per_trial,
        size=cfg.size,
        max_steps=cfg.max_steps,
        step_penalty=cfg.step_penalty,
        success_reward=cfg.success_reward,
        wrong_terminal_reward=cfg.wrong_terminal_reward,
        object_types=cfg.object_types,
        object_colors=cfg.object_colors,
        max_depth=cfg.max_depth,
        prune_prob=cfg.prune_prob,
        num_distractor_rules=cfg.num_distractor_rules,
        num_distractor_objects=cfg.num_distractor_objects,
    )


@torch.no_grad()
def evaluate(model: RecurrentActorCritic, cfg: PPOConfig, device: torch.device) -> dict:
    env = make_env(cfg)
    model.eval()

    trial_returns = []
    episode_returns = [[] for _ in range(cfg.episodes_per_trial)]

    for trial_idx in range(cfg.eval_trials):
        obs, _ = env.reset(seed=10_000 + trial_idx)
        state = model.initial_state(device)
        done = False
        trial_return = 0.0
        current_episode = 0
        running_ep_return = 0.0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _, _, state = model.get_action_and_value(obs_t, state)
            obs, reward, terminated, truncated, info = env.step(int(action.item()))
            done = terminated or truncated

            reward = float(reward)
            trial_return += reward
            running_ep_return += reward

            new_episode = info.get("trial_episode", current_episode)
            if new_episode != current_episode:
                if current_episode < cfg.episodes_per_trial:
                    episode_returns[current_episode].append(running_ep_return)
                running_ep_return = 0.0
                current_episode = new_episode

        if current_episode < cfg.episodes_per_trial:
            episode_returns[current_episode].append(running_ep_return)

        trial_returns.append(trial_return)

    model.train()

    per_episode_means = [float(np.mean(x)) if x else 0.0 for x in episode_returns]
    return {
        "mean_trial_return": float(np.mean(trial_returns)) if trial_returns else 0.0,
        "p20_trial_return": float(np.percentile(trial_returns, 20)) if trial_returns else 0.0,
        "per_episode_means": per_episode_means,
    }


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="RL^2 + PPO with an LSTM policy")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--exp-name", type=str, default="rl2_ppo")
    parser.add_argument("--save-every-updates", type=int, default=25)
    parser.add_argument("--eval-every-updates", type=int, default=10)
    parser.add_argument("--eval-trials", type=int, default=20)

    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=75)
    parser.add_argument("--step-penalty", type=float, default=0.0)
    parser.add_argument("--success-reward", type=float, default=1.0)
    parser.add_argument("--wrong-terminal-reward", type=float, default=0.0)
    parser.add_argument("--object-types", type=int, default=4)
    parser.add_argument("--object-colors", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--prune-prob", type=float, default=0.1)
    parser.add_argument("--num-distractor-rules", type=int, default=0)
    parser.add_argument("--num-distractor-objects", type=int, default=0)
    parser.add_argument("--episodes-per-trial", type=int, default=4)

    args = parser.parse_args()
    return PPOConfig(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_size=args.hidden_size,
        seed=args.seed,
        device=args.device,
        results_dir=args.results_dir,
        exp_name=args.exp_name,
        save_every_updates=args.save_every_updates,
        eval_every_updates=args.eval_every_updates,
        eval_trials=args.eval_trials,
        size=args.size,
        max_steps=args.max_steps,
        step_penalty=args.step_penalty,
        success_reward=args.success_reward,
        wrong_terminal_reward=args.wrong_terminal_reward,
        object_types=args.object_types,
        object_colors=args.object_colors,
        max_depth=args.max_depth,
        prune_prob=args.prune_prob,
        num_distractor_rules=args.num_distractor_rules,
        num_distractor_objects=args.num_distractor_objects,
        episodes_per_trial=args.episodes_per_trial,
    )


def write_metrics_header(path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "update",
            "global_step",
            "avg_rollout_reward",
            "avg_completed_trial_return",
            "policy_loss",
            "value_loss",
            "entropy",
            "eval_mean_trial_return",
            "eval_p20_trial_return",
            "eval_ep0_mean_return",
            "eval_ep1_mean_return",
            "eval_ep2_mean_return",
            "eval_ep3_mean_return",
            "elapsed_sec",
        ])


def append_metrics(path: str, row: list) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    ensure_dir(cfg.results_dir)
    ckpt_dir = os.path.join(cfg.results_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    env = make_env(cfg)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = RecurrentActorCritic(obs_dim, action_dim, cfg.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    num_updates = max(1, cfg.total_timesteps // cfg.rollout_steps)
    metrics_path = os.path.join(cfg.results_dir, f"{cfg.exp_name}_metrics.csv")
    write_metrics_header(metrics_path)

    obs, _ = env.reset(seed=cfg.seed)
    state = model.initial_state(device)
    global_step = 0
    completed_trial_returns = []
    running_trial_return = 0.0
    start_time = time.time()

    print(f"Starting RL^2 PPO training for {num_updates} updates.", flush=True)

    for update in range(1, num_updates + 1):
        obs_buf = []
        actions_buf = []
        logprobs_buf = []
        rewards_buf = []
        values_buf = []
        dones_buf = []
        rollout_initial_state = (state[0].detach().clone(), state[1].detach().clone())

        for _ in range(cfg.rollout_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, logprob, _, value, next_state = model.get_action_and_value(obs_t, state)

            obs_buf.append(obs_t.squeeze(0))
            actions_buf.append(action.squeeze(0) if action.ndim > 0 else action)
            logprobs_buf.append(logprob.squeeze(0) if logprob.ndim > 0 else logprob)
            values_buf.append(value.squeeze(0) if value.ndim > 0 else value)

            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            rewards_buf.append(torch.tensor(float(reward), dtype=torch.float32, device=device))
            dones_buf.append(torch.tensor(float(done), dtype=torch.float32, device=device))

            running_trial_return += float(reward)
            global_step += 1
            obs = next_obs
            state = next_state

            if done:
                completed_trial_returns.append(running_trial_return)
                running_trial_return = 0.0
                obs, _ = env.reset()
                state = model.initial_state(device)

        with torch.no_grad():
            next_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, next_value, _ = model.forward_step(next_obs_t, state)
            next_value = next_value.squeeze(0)

        rewards_t = torch.stack(rewards_buf)
        values_t = torch.stack(values_buf).detach()
        dones_t = torch.stack(dones_buf)
        obs_t = torch.stack(obs_buf)
        actions_t = torch.stack(actions_buf).long()
        old_logprobs_t = torch.stack(logprobs_buf).detach()

        advantages = torch.zeros_like(rewards_t, device=device)
        lastgaelam = torch.tensor(0.0, device=device)

        for t in reversed(range(len(rewards_t))):
            if t == len(rewards_t) - 1:
                nextvalues = next_value
                nextnonterminal = 1.0 - dones_t[t]
            else:
                nextvalues = values_t[t + 1]
                nextnonterminal = 1.0 - dones_t[t]
            delta = rewards_t[t] + cfg.gamma * nextvalues * nextnonterminal - values_t[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns_t = (advantages + values_t).detach()
        advantages_t = (
            (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        ).detach()

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0

        for _ in range(cfg.update_epochs):
            sequence_initial_state = (
                rollout_initial_state[0].detach(),
                rollout_initial_state[1].detach(),
            )
            new_logprobs_t, entropies_t, new_values_t = model.evaluate_sequence(
                obs_t,
                actions_t,
                dones_t,
                sequence_initial_state,
            )
            ratio = (new_logprobs_t - old_logprobs_t).exp()

            pg_loss1 = -advantages_t * ratio
            pg_loss2 = -advantages_t * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()

            value_loss_unclipped = (new_values_t - returns_t) ** 2
            value_clipped = values_t + torch.clamp(new_values_t - values_t, -cfg.clip_coef, cfg.clip_coef)
            value_loss_clipped = (value_clipped - returns_t) ** 2
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            entropy_loss = entropies_t.mean()
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_entropy = float(entropy_loss.item())

        eval_mean = float("nan")
        eval_p20 = float("nan")
        eval_ep_means = [float("nan")] * cfg.episodes_per_trial

        if update == 1 or update % cfg.eval_every_updates == 0 or update == num_updates:
            eval_out = evaluate(model, cfg, device)
            eval_mean = eval_out["mean_trial_return"]
            eval_p20 = eval_out["p20_trial_return"]
            eval_ep_means = eval_out["per_episode_means"]

        if update % cfg.save_every_updates == 0 or update == num_updates:
            ckpt_path = os.path.join(ckpt_dir, f"{cfg.exp_name}_update{update}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "update": update,
                    "global_step": global_step,
                },
                ckpt_path,
            )

        avg_rollout_reward = float(torch.stack(rewards_buf).mean().item())
        avg_completed_trial_return = float(np.mean(completed_trial_returns[-20:])) if completed_trial_returns else 0.0
        elapsed = time.time() - start_time

        metrics_row = [
            update,
            global_step,
            avg_rollout_reward,
            avg_completed_trial_return,
            last_policy_loss,
            last_value_loss,
            last_entropy,
            eval_mean,
            eval_p20,
        ]
        for i in range(4):
            metrics_row.append(eval_ep_means[i] if i < len(eval_ep_means) else float("nan"))
        metrics_row.append(elapsed)
        append_metrics(metrics_path, metrics_row)

        print(
            f"update={update}/{num_updates} "
            f"step={global_step} "
            f"avg_rollout_reward={avg_rollout_reward:.4f} "
            f"avg_trial_return={avg_completed_trial_return:.4f} "
            f"policy_loss={last_policy_loss:.4f} "
            f"value_loss={last_value_loss:.4f} "
            f"entropy={last_entropy:.4f} "
            f"eval_mean={eval_mean:.4f} "
            f"eval_p20={eval_p20:.4f}",
            flush=True,
        )

    final_ckpt = os.path.join(ckpt_dir, f"{cfg.exp_name}_final.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "global_step": global_step,
        },
        final_ckpt,
    )
    print(f"Saved final checkpoint to {final_ckpt}", flush=True)
    print(f"Saved metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
