"""Run a trained HIL-SERL SAC policy on the real SO101 in inference-only mode.

LeRobot's `lerobot.rl.eval_policy` is broken (references a non-existent
`env_cfg.pretrained_policy_name_or_path` field) and `lerobot.rl.actor`
is coupled to a gRPC learner — neither works for standalone SAC inference.
This script wraps the same `make_robot_env` + `make_policy` flow without
gRPC, and without learner-side updates.

Usage:

    uv run python scripts/eval_sac_so101.py \\
        --config_path=configs/train_config_hilserl_so101.json \\
        --policy.path=outputs/train/so101_hilserl/checkpoints/last/pretrained_model

Optional env var:
    EVAL_N_EPISODES=10  (default 5)
"""

import logging
import os

import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy  # noqa: F401  ensure SAC class is imported
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.rl.gym_manipulator import make_robot_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_sac_so101")


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig) -> None:
    # parser.wrap doesn't auto-call validate(), so --policy.path doesn't yet
    # propagate to cfg.policy.pretrained_path. Trigger it explicitly.
    cfg.validate()

    if cfg.policy is None or cfg.policy.pretrained_path is None:
        raise ValueError(
            "Missing --policy.path. Pass the trained SAC checkpoint dir, e.g. "
            "--policy.path=outputs/train/so101_hilserl/checkpoints/last/pretrained_model"
        )

    log.info("Building env (real SO101 robot via gym_manipulator pipeline)...")
    env, _teleop = make_robot_env(cfg=cfg.env)

    log.info(f"Loading SAC policy from {cfg.policy.pretrained_path}")
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.eval()

    n_episodes = int(os.environ.get("EVAL_N_EPISODES", 5))
    log.info(f"Running {n_episodes} episodes (override via EVAL_N_EPISODES env var)")

    successes = 0
    for ep in range(n_episodes):
        obs, _info = env.reset()
        ep_reward = 0.0
        steps = 0
        while True:
            with torch.no_grad():
                obs_filtered = {
                    k: v for k, v in obs.items() if k in cfg.policy.input_features
                }
                action = policy.select_action(batch=obs_filtered)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            steps += 1
            if terminated or truncated:
                break
        if ep_reward > 0:
            successes += 1
        log.info(f"Episode {ep + 1}/{n_episodes}: {steps} steps, reward={ep_reward:.2f}")

    log.info(f"Success rate: {successes}/{n_episodes} ({100 * successes / n_episodes:.0f}%)")


if __name__ == "__main__":
    main()
