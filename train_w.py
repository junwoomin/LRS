import os
import time
import math
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
import numpy as np
import torch
import cv2
import logging

from loop import BEVPathFollowEnv, RewardConfig

from roach.ppo import PPOAgent, PPOConfig, RolloutBuffer
from collections import Counter

try:
    import wandb
except ImportError:
    wandb = None

def init_wandb(algo: str, town: str, total_timesteps: int, extra_cfg: dict):
    use_wandb = bool(int(os.environ.get("WANDB", "0")))
    if not use_wandb:
        return None
    if wandb is None:
        raise RuntimeError("WANDB=1 requires wandb. Install it with: pip install wandb")

    run_name = (
        f"{algo}-{town}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "LRS"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=run_name,
        config={
            "algo": algo,
            "town": town,
            "total_timesteps": total_timesteps,
            **extra_cfg,
        },
    )

    # x-axis를 global_step으로 통일 (차트 보기 편함)
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("episode/*", step_metric="global_step")
    wandb.define_metric("perf/*", step_metric="global_step")

    return run
def get_clean_reason(reason_str):
    if reason_str is None: return "Unknown"
    if "stationary" in reason_str: return "Stationary"
    if "traffic light" in reason_str: return "Traffic Light"
    if "veh_collision" in reason_str: return "Vehicle Collision"
    if "DAS_collision" in reason_str: return "DAS Collision"
    if "off route" in reason_str: return "Off Route"
    if "success" in reason_str: return "Success"
    if "max steps" in reason_str: return "Timeout"
    return "Other"

def get_fail_reason(env, info: dict) -> str:
    # env.reason 우선, 없으면 info에서 탐색
    reason = getattr(env, "reason", None)
    if reason is None and isinstance(info, dict):
        reason = info.get("reason", None) or info.get("failure_reason", None)
    if reason is None:
        reason = "unknown"
    return str(reason)

def get_progress_percent(env, info: dict) -> float:
    """
    env.path_idx / env.ego_path 기반 진행률을 0~100(%)로 반환.
    - ego_path가 list/np.array이면 len 기반으로 계산
    - ego_path가 숫자면 분모로 사용
    - info에 이미 progress/percent가 있으면 그걸 우선 사용(있다면)
    """
    if isinstance(info, dict):
        for k in ["progress_percent", "progress_pct", "route_completion", "progress"]:
            if k in info:
                try:
                    v = float(info[k])
                    # 이미 0~100일 수도, 0~1일 수도 있음
                    if v <= 1.0:
                        return float(np.clip(v * 100.0, 0.0, 100.0))
                    return float(np.clip(v, 0.0, 100.0))
                except:
                    pass

    path_idx = getattr(env, "path_idx", None)
    ego_path = getattr(env, "ego_path", None)

    if path_idx is None or ego_path is None:
        return float("nan")

    try:
        # ego_path가 시퀀스(경로 점들)인 경우
        if hasattr(ego_path, "__len__") and not isinstance(ego_path, (str, bytes)):
            denom = max(1, len(ego_path) - 1)
            pct = (float(path_idx) / float(denom)) * 100.0
            return float(np.clip(pct, 0.0, 100.0))

        # ego_path가 숫자(분모)인 경우
        denom = float(ego_path)
        if denom <= 0:
            return float("nan")
        pct = (float(path_idx) / denom) * 100.0
        return float(np.clip(pct, 0.0, 100.0))
    except:
        return float("nan")




def is_gymnasium_step(step_out) -> bool:
    # Gymnasium: obs, reward, terminated, truncated, info (5-tuple)
    return isinstance(step_out, tuple) and len(step_out) == 5


def reset_env(env, seed=None):
    out = env.reset(seed=seed)
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        obs, info = out, {}
    return obs, info


def step_env(env, action):
    out = env.step(action)
    if is_gymnasium_step(out):
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        # info에 terminated/truncated 넣어두면 패널에서 보기 편함
        if isinstance(info, dict):
            info = dict(info)
            info["terminated"] = bool(terminated)
            info["truncated"] = bool(truncated)
        return obs, reward, done, info
    else:
        obs, reward, done, info = out
        return obs, reward, bool(done), info


def bev_to_overlay_bgr(bev: np.ndarray, out_size: int = 400) -> np.ndarray:

    out = cv2.resize(bev, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return out

def info_to_panel_bgr(info: Dict[str, Any], height: int = 400, width: int = 320) -> np.ndarray:
    """
    오른쪽 패널에 info 텍스트를 그려줍니다.
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    y = 28
    line_h = 22

    def put(line: str):
        nonlocal y
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_h

    # key ordering (원하면 여기서 더 추가/정렬 가능)
    keys = [
        "step", "path_idx", "success", "collision",
        "speed", "target_v", "cte",
        "terminated", "truncated"
    ]

    put("INFO")
    put("-" * 24)
    for k in keys:
        if k in info:
            v = info[k]
            if isinstance(v, float):
                put(f"{k}: {v:.3f}")
            else:
                put(f"{k}: {v}")

    # extra keys (너무 많으면 자르기)
    extras = [k for k in info.keys() if k not in keys]
    if extras:
        y += 10
        put("EXTRA")
        put("-" * 24)
        for k in extras[:12]:
            v = info[k]
            if isinstance(v, float):
                put(f"{k}: {v:.3f}")
            else:
                put(f"{k}: {v}")

    return panel


def open_video_writer_with_fallback(base_path_no_ext: str, fps: int, size_wh: Tuple[int, int]) -> Tuple[cv2.VideoWriter, str]:
    candidates = [
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("H264", ".mp4"),
        ("XVID", ".avi"),
        ("MJPG", ".avi"),
    ]

    base = Path(base_path_no_ext)
    base.parent.mkdir(parents=True, exist_ok=True)

    for fourcc_str, ext in candidates:
        out_path = str(base.with_suffix(ext))
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        vw = cv2.VideoWriter(out_path, fourcc, float(fps), size_wh)

        if vw.isOpened():
            print(f"[Video] opened codec={fourcc_str}, path={Path(out_path).resolve()}")
            return vw, out_path

        vw.release()

    raise RuntimeError(
        f"Failed to open VideoWriter. Tried codecs={candidates}. "
        f"base={Path(base_path_no_ext).resolve()} size={size_wh} fps={fps}"
    )

def train(
    env,
    town: str,
    vis: bool,
    render_mode: str,
    total_timesteps: int,
    save_every_steps: int,
    log_every_steps: int,
    video_fps: int,
    video_bev_resize: int,
    panel_w: int,
    video_steps: int,
    wandb_run=None,
):
    # -------------------------
    # init obs / shapes
    # -------------------------
    if hasattr(env, "set_curriculum"):
        env.set_curriculum(0.0)

    obs, info = reset_env(env, seed=0)

    bev_shape = tuple(obs["birdview"].shape)
    state_dim = int(np.asarray(obs["state"]).shape[0])
    action_dim = int(np.prod(env.action_space.shape))

    cfg = PPOConfig(
        rollout_steps=int(os.environ.get("ROLLOUT_STEPS", "2048")),
        update_epochs=int(os.environ.get("PPO_UPDATE_EPOCHS", "20")),
        minibatch_size=int(os.environ.get("PPO_MINIBATCH_SIZE", "256")),
        clip_range=float(os.environ.get("PPO_CLIP_RANGE", "0.2")),
        lr=float(os.environ.get("PPO_LR", "1e-5")),
        max_grad_norm=float(os.environ.get("PPO_MAX_GRAD_NORM", "0.5")),
        gamma=float(os.environ.get("PPO_GAMMA", "0.99")),
        gae_lambda=float(os.environ.get("PPO_GAE_LAMBDA", "0.9")),
        clip_eps=float(os.environ.get("PPO_CLIP_EPS", "0.2")),
        ent_coef=float(os.environ.get("PPO_ENT_COEF", "0.01")),
        vf_coef=float(os.environ.get("PPO_VF_COEF", "0.5")),
    )

    agent = PPOAgent(
        bev_shape=bev_shape,
        state_dim=state_dim,
        action_dim=action_dim,
        cfg=cfg,
    )

    agent.load(env.checkpoint_path)

    num_updates = max(1, int(np.ceil(total_timesteps / cfg.rollout_steps)))
    agent.set_scheduler(num_updates)

    # Reuse the checkpoint policy already loaded for NPC inference.
    # The original rollout/PPO policy connection is retained; see README limitations.
    _policy = env._policy
    rollout = RolloutBuffer(
        size=cfg.rollout_steps,
        bev_shape=bev_shape,
        state_dim=state_dim,
        action_dim=action_dim,
        device=env.device,
    )

    # -------------------------
    # stats
    # -------------------------
    global_step = 0
    update_idx = 0
    episode_idx = 0

    episode_return = 0.0
    episode_len = 0
    episode_action_0 = 1e-8
    episode_action_1 = 1e-8

    ep_returns = []
    ep_lens = []
    last_stats = None
    reason_counts = Counter()

    t0 = time.time()

    # -------------------------
    # video states
    # -------------------------
    vw = None
    video_out_path = None
    recording_left = 0
    last_video_start_step = -10**18
    current_video_artifact_dir = None
    pending_video_wandb_save = None

    # -------------------------
    # safe train loop
    # -------------------------
    try:
        while global_step < total_timesteps:
            rollout.reset()

            while rollout.ptr < cfg.rollout_steps and global_step < total_timesteps:
                if hasattr(env, "set_curriculum"):
                    env.set_curriculum(global_step / max(1, total_timesteps))

                global_step += 1

                # ---------------------------------
                # start periodic recording
                # ---------------------------------
                start_record = (
                    (global_step % save_every_steps == 0)
                    or (global_step == total_timesteps)
                )

                if start_record and (vw is None) and (global_step - last_video_start_step >= save_every_steps):
                    current_video_artifact_dir = Path("artifacts") / town / f"s{global_step:07d}"
                    current_video_artifact_dir.mkdir(parents=True, exist_ok=True)

                    base_no_ext = str(current_video_artifact_dir / f"rollout_{town}_s{global_step:07d}")
                    H = int(video_bev_resize)
                    W = int(video_bev_resize) + int(panel_w)

                    vw, video_out_path = open_video_writer_with_fallback(
                        base_no_ext,
                        fps=video_fps,
                        size_wh=(W, H),
                    )

                    if vw is None or (hasattr(vw, "isOpened") and not vw.isOpened()):
                        raise RuntimeError(f"Failed to open video writer: {base_no_ext}")

                    recording_left = int(video_steps)
                    last_video_start_step = global_step
                    pending_video_wandb_save = None

                # ---------------------------------
                # act
                # ---------------------------------
                # action, logp, value = agent.act(obs, deterministic=True)
                obs_input = dict([(k, torch.as_tensor(v).to(env.device).unsqueeze(0)) for k, v in obs.items()])

                action, value, logp, mu, sigma, features,cnn_feature = _policy.forward(
                    obs_input, deterministic=True, clip_action=True)

                if isinstance(action, torch.Tensor):
                    action = action.detach().cpu().numpy()
                action = np.asarray(action, dtype=np.float32)

                # ---------------------------------
                # env step
                # ---------------------------------
                next_obs, reward, done, info = step_env(env, action)

                # ---------------------------------
                # rollout store
                # ---------------------------------
                rollout.add(
                    obs=obs,
                    action=action,
                    logp=float(logp),
                    value=float(value),
                    reward=float(reward),
                    done=bool(done),
                )

                # ---------------------------------
                # episode stats
                # ---------------------------------
                episode_return += float(reward)
                episode_len += 1
                episode_action_0 += float(info.get("action_0", 0.0))
                episode_action_1 += float(info.get("action_1", 0.0))

                # ---------------------------------
                # optional live render
                # ---------------------------------
                if vis:
                    env.render()

                # ---------------------------------
                # video write
                # ---------------------------------
                if vw is not None and recording_left > 0:
                    bev = next_obs.get("bev", None)
                    if bev is not None:
                        if isinstance(bev, torch.Tensor):
                            bev = bev.detach().cpu().numpy()
                        if isinstance(bev, np.ndarray) and bev.ndim == 4:
                            bev = bev[0]

                        info_vis = dict(info) if isinstance(info, dict) else {}
                        info_vis.update({
                            "algo": "ppo",
                            "global_step": int(global_step),
                            "reward": float(reward),
                            "done": bool(done),
                            "update_idx": int(update_idx),
                        })

                        bev_img = bev_to_overlay_bgr(bev, out_size=video_bev_resize)
                        panel = info_to_panel_bgr(
                            info_vis,
                            height=video_bev_resize,
                            width=panel_w,
                        )
                        frame = np.concatenate([bev_img, panel], axis=1)
                        frame = np.ascontiguousarray(frame.astype(np.uint8))

                        expected_h = int(video_bev_resize)
                        expected_w = int(video_bev_resize) + int(panel_w)

                        if not (frame.ndim == 3 and frame.shape[2] == 3):
                            raise RuntimeError(f"Invalid video frame shape: {frame.shape}")
                        if frame.shape[0] != expected_h or frame.shape[1] != expected_w:
                            raise RuntimeError(
                                f"Video frame size mismatch: frame={frame.shape}, expected=({expected_h}, {expected_w}, 3)"
                            )

                        vw.write(frame)

                    recording_left -= 1

                    if recording_left <= 0:
                        vw.release()
                        print(f"Saved video: {Path(video_out_path).resolve() if video_out_path else video_out_path}")
                        pending_video_wandb_save = video_out_path
                        vw = None
                        video_out_path = None
                        current_video_artifact_dir = None

                # ---------------------------------
                # episode done handling
                # ---------------------------------
                if done:
                    episode_idx += 1

                    ep_ret = float(episode_return)
                    ep_len = int(episode_len)

                    fail_reason = get_fail_reason(env, info)
                    clean_reason = get_clean_reason(fail_reason)
                    reason_counts[clean_reason] += 1

                    progress_pct = get_progress_percent(env, info if isinstance(info, dict) else {})

                    if wandb_run is not None:
                        wandb.log(
                            {
                                "global_step": int(global_step),
                                "episode/index": int(episode_idx),
                                "episode/return": ep_ret,
                                "episode/len": ep_len,
                                "episode/action_0": float(episode_action_0 / max(1, episode_len)),
                                "episode/action_1": float(episode_action_1 / max(1, episode_len)),
                                "episode/progress_percent": float(progress_pct),
                                "perf/fps": float(int(global_step / max(1e-6, (time.time() - t0)))),
                            },
                            step=int(global_step),
                        )

                        data = [[reason, count] for reason, count in reason_counts.items()]
                        table = wandb.Table(data=data, columns=["Reason", "Count"])
                        wandb.log({
                            "charts/fail_reason_distribution": wandb.plot.bar(
                                table,
                                "Reason",
                                "Count",
                                title="Fail Reason Distribution",
                            )
                        }, step=int(global_step))

                    ep_returns.append(ep_ret)
                    ep_lens.append(ep_len)

                    episode_return = 0.0
                    episode_len = 0
                    episode_action_0 = 1e-8
                    episode_action_1 = 1e-8

                    if hasattr(env, "set_curriculum"):
                        env.set_curriculum(global_step / max(1, total_timesteps))
                    next_obs, _ = reset_env(env, seed=None)

                obs = next_obs

                # ---------------------------------
                # periodic logging
                # ---------------------------------
                if global_step % log_every_steps == 0:
                    elapsed = time.time() - t0
                    fps = int(global_step / max(1e-6, elapsed))
                    avg_ret = float(np.mean(ep_returns[-10:])) if ep_returns else 0.0
                    avg_len = float(np.mean(ep_lens[-10:])) if ep_lens else 0.0

                    if last_stats is None:
                        print(
                            f"[ppo step={global_step}] "
                            f"fps={fps} avg_return(10)={avg_ret:.2f} avg_len(10)={avg_len:.1f}"
                        )
                    else:
                        print(
                            f"[ppo step={global_step}] fps={fps} "
                            f"avg_return(10)={avg_ret:.2f} avg_len(10)={avg_len:.1f} "
                            f"loss_pi={last_stats.get('loss_pi', 0.0):.4f} "
                            f"loss_v={last_stats.get('loss_v', 0.0):.4f} "
                            f"entropy={last_stats.get('entropy', 0.0):.4f} "
                            f"approx_kl={last_stats.get('approx_kl', 0.0):.4f}"
                        )

                    if wandb_run is not None:
                        payload = {
                            "global_step": int(global_step),
                            "perf/fps": float(fps),
                            "episode/avg_return_10": avg_ret,
                            "episode/avg_len_10": avg_len,
                        }
                        if last_stats is not None:
                            payload.update({
                                "train/loss_pi": float(last_stats.get("loss_pi", 0.0)),
                                "train/loss_v": float(last_stats.get("loss_v", 0.0)),
                                "train/entropy": float(last_stats.get("entropy", 0.0)),
                                "train/approx_kl": float(last_stats.get("approx_kl", 0.0)),
                                "train/lr": float(last_stats.get("lr", 0.0)),
                            })
                        wandb.log(payload, step=int(global_step))

                # ---------------------------------
                # periodic checkpoint save
                # ---------------------------------
                if global_step % save_every_steps == 0 or global_step == total_timesteps:
                    artifact_dir = Path("roach_run") / town / f"s{global_step:07d}"
                    artifact_dir.mkdir(parents=True, exist_ok=True)

                    ckpt_path = artifact_dir / f"ppo_{town}_s{global_step:07d}.pth"
                    agent.save(str(ckpt_path))
                    print(f"Saved checkpoint: {ckpt_path.resolve()}")

                    if wandb_run is not None:
                        wandb.save(str(ckpt_path))

                # ---------------------------------
                # if a video finished this step, upload after finalize
                # ---------------------------------
                if pending_video_wandb_save is not None and wandb_run is not None:
                    wandb.save(str(pending_video_wandb_save))
                    pending_video_wandb_save = None

            # ---------------------------------
            # PPO update after rollout
            # ---------------------------------
            if rollout.ptr > 0:
                last_value = 0.0 if done else agent.value(obs)
                rollout.compute_gae(
                    last_value=last_value,
                    gamma=cfg.gamma,
                    lam=cfg.gae_lambda,
                )
                last_stats = agent.update(rollout)
                update_idx += 1

                if wandb_run is not None:
                    wandb.log({
                        "global_step": int(global_step),
                        "update/index": int(update_idx),
                        "train/loss_pi": float(last_stats.get("loss_pi", 0.0)),
                        "train/loss_v": float(last_stats.get("loss_v", 0.0)),
                        "train/entropy": float(last_stats.get("entropy", 0.0)),
                        "train/approx_kl": float(last_stats.get("approx_kl", 0.0)),
                        "train/lr": float(last_stats.get("lr", 0.0)),
                    }, step=int(global_step))

    finally:
        if vw is not None:
            vw.release()
            print(f"Finalized video: {Path(video_out_path).resolve() if video_out_path else video_out_path}")
            if wandb_run is not None and video_out_path is not None:
                wandb.save(str(video_out_path))
def main():
    os.makedirs("roach_run/checkpoints", exist_ok=True)

    logging.basicConfig(
        filename=os.path.join("roach/results.log"),
        filemode='w',
        format='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    start_time_raw = time.time()
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"\n" + "=" * 50)
    logger.info(f"🚀 학습 시작 시간: {start_time_str}")
    logger.info("=" * 50 + "\n")

    town = os.environ.get("TOWN", "Town03")
    vis = bool(int(os.environ.get("VIS", "1")))
    render_mode = "human" if vis else "none"

    total_timesteps = int(os.environ.get("TOTAL_TIMESTEPS", "5000000"))
    rollout_steps = int(os.environ.get("ROLLOUT_STEPS", "2048"))

    save_every_updates = int(os.environ.get("SAVE_EVERY_UPDATES", "20"))
    save_every_steps = int(os.environ.get("PPO_SAVE_EVERY_STEPS", str(save_every_updates * rollout_steps)))
    log_every_steps = int(os.environ.get("PPO_LOG_EVERY_STEPS", "2000"))
    video_steps = int(os.environ.get("PPO_VIDEO_STEPS", "600"))

    video_fps = int(os.environ.get("VIDEO_FPS", "12"))
    video_bev_resize = int(os.environ.get("VIDEO_BEV_SIZE", "400"))
    panel_w = int(os.environ.get("VIDEO_PANEL_W", "320"))

    reward_cfg = RewardConfig()

    CONTROL_FPS = 30
    SUBSTEPS = 1
    PHYSICS_FPS = CONTROL_FPS * SUBSTEPS
    FIXED_DT = 1.0 / PHYSICS_FPS

    env = BEVPathFollowEnv(
        town=town,
        vis=vis,
        render_mode=render_mode,
        out_size=192,
        panel_update_every=1,
        render_fps=PHYSICS_FPS,
        sim_steps_per_frame=SUBSTEPS,
        FIXED_DT=FIXED_DT,
        max_steps=2000,
        reward_cfg=reward_cfg,
        logger=logger
    )

    extra_cfg = {
        "algo": "ppo",
        "rollout_steps": rollout_steps,
        "save_every_steps": save_every_steps,
        "log_every_steps": log_every_steps,
        "video_steps": video_steps,
        "video_fps": video_fps,
        "ppo_lr": float(os.environ.get("PPO_LR", "1e-5")),
        "ppo_epochs": int(os.environ.get("PPO_UPDATE_EPOCHS", "20")),
        "ppo_minibatch_size": int(os.environ.get("PPO_MINIBATCH_SIZE", "256")),
        "ppo_gamma": float(os.environ.get("PPO_GAMMA", "0.99")),
        "ppo_gae_lambda": float(os.environ.get("PPO_GAE_LAMBDA", "0.9")),
        "ppo_ent_coef": float(os.environ.get("PPO_ENT_COEF", "0.01")),
        "ppo_vf_coef": float(os.environ.get("PPO_VF_COEF", "0.5")),
    }

    run = init_wandb(
        algo='roach',
        town=town,
        total_timesteps=total_timesteps,
        extra_cfg=extra_cfg
    )

    train(
        env=env,
        town=town,
        vis=vis,
        render_mode=render_mode,
        total_timesteps=total_timesteps,
        save_every_steps=save_every_steps,
        log_every_steps=log_every_steps,
        video_fps=video_fps,
        video_bev_resize=video_bev_resize,
        panel_w=panel_w,
        video_steps=video_steps,
        wandb_run=run
    )

    env.close()

    end_time_raw = time.time()
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    elapsed_seconds = int(end_time_raw - start_time_raw)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info("\n" + "=" * 50)
    logger.info(f"🏁 학습 종료 시간: {end_time_str}")
    logger.info(f"⏱️ 총 소요 시간: {hours}시간 {minutes}분 {seconds}초")
    logger.info("=" * 50 + "\n")


if __name__ == "__main__":
    main()
