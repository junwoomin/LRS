import json
import time
from pathlib import Path
import numpy as np
import cv2
from stable_baselines3.common.callbacks import BaseCallback
from omegaconf import OmegaConf
from local_logging import LocalRun


def save_video(path, frames, fps=30):
    if not frames:
        return
    height, width = frames[0].shape[:2]
    encoder = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not encoder.isOpened():
        raise RuntimeError(f'Cannot open video writer: {path}')
    try:
        for frame in frames:
            encoder.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        encoder.release()


class LocalCallback(BaseCallback):
    """Roach callback preserving evaluation, checkpoints, videos and local metrics."""
    def __init__(self, cfg, vec_env, output_dir='roach_run/callback'):
        super().__init__(verbose=1)
        config = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
        self.run = LocalRun(output_dir, config)
        self._video_path = self.run.directory / 'video'
        self._ckpt_dir = self.run.directory / 'ckpt'
        self._video_path.mkdir(parents=True, exist_ok=True)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vec_env = vec_env
        self._eval_step = int(1e5)
        self._buffer_step = int(1e5)

    def _init_callback(self):
        self.n_epoch = 0
        self._last_time_buffer = self.model.num_timesteps
        self._last_time_eval = self.model.num_timesteps

    def _on_step(self):
        return True

    def _on_training_end(self):
        print(f'n_epoch: {self.n_epoch}, num_timesteps: {self.model.num_timesteps}')
        elapsed = max(time.time() - self.model.start_time, 1e-6)
        self.run.log({
            'time/n_epoch': self.n_epoch,
            'time/sec_per_epoch': elapsed / (self.n_epoch + 1),
            'time/fps': (self.model.num_timesteps - self.model.start_num_timesteps) / elapsed,
            'time/train': self.model.t_train,
            'time/train_values': self.model.t_train_values,
            'time/rollout': self.model.t_rollout,
            **self.model.train_debug,
        }, step=self.model.num_timesteps)
        if self.model.num_timesteps - self._last_time_eval >= self._eval_step:
            self._last_time_eval = self.model.num_timesteps
            video_path = self._video_path / f'eval_{self.model.num_timesteps}.mp4'
            stats, events = self.evaluate_policy(self.vec_env, self.model.policy, str(video_path))
            self.run.log({**stats, 'video/eval': str(video_path)}, step=self.model.num_timesteps)
            (self.run.directory / f'events_{self.model.num_timesteps}.json').write_text(json.dumps(events, default=lambda v: np.asarray(v).tolist()))
            self.model.save(str(self._ckpt_dir / f'ckpt_{self.model.num_timesteps}.pth'))
        self.n_epoch += 1

    def _on_rollout_end(self):
        stats = self.get_avg_ep_stat(self.model.ep_stat_buffer, prefix='rollout/')
        stats['time/rollout'] = self.model.t_rollout
        for name, values in [('action', self.model.action_statistics), ('alpha', self.model.mu_statistics), ('beta', self.model.sigma_statistics)]:
            values = np.asarray(values)
            if values.size:
                values = values.reshape(-1, values.shape[-1])
                for i in range(values.shape[-1]):
                    counts, edges = np.histogram(values[:, i], bins=64)
                    stats[f'{name}[{i}]'] = {'counts': counts, 'edges': edges}
        self.run.log(stats, step=self.model.num_timesteps)
        if self.model.num_timesteps - self._last_time_buffer >= self._buffer_step:
            self._last_time_buffer = self.model.num_timesteps
            path = self._video_path / f'buffer_{self.model.num_timesteps}.mp4'
            save_video(path, self.model.buffer.render())
            self.run.log({'video/buffer': str(path)}, step=self.model.num_timesteps)

    @staticmethod
    def evaluate_policy(env, policy, video_path, min_eval_steps=3000):
        policy = policy.eval()
        t0 = time.time()
        for i in range(env.num_envs):
            env.set_attr('eval_mode', True, indices=i)
        obs = env.reset()

        list_render = []
        ep_stat_buffer = []
        ep_events = {}
        for i in range(env.num_envs):
            ep_events[f'venv_{i}'] = []

        n_step = 0
        n_timeout = 0
        env_done = np.array([False]*env.num_envs)
        # while n_step < min_eval_steps:
        while n_step < min_eval_steps or not np.all(env_done):
            actions, values, log_probs, mu, sigma, _ = policy.forward(obs, deterministic=True, clip_action=True)
            obs, reward, done, info = env.step(actions)

            for i in range(env.num_envs):
                env.set_attr('action_value', values[i], indices=i)
                env.set_attr('action_log_probs', log_probs[i], indices=i)
                env.set_attr('action_mu', mu[i], indices=i)
                env.set_attr('action_sigma', sigma[i], indices=i)

            list_render.append(env.render(mode='rgb_array'))

            n_step += 1
            env_done |= done

            for i in np.where(done)[0]:
                ep_stat_buffer.append(info[i]['episode_stat'])
                ep_events[f'venv_{i}'].append(info[i]['episode_event'])
                n_timeout += int(info[i]['timeout'])

        # conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
        save_video(video_path, list_render)

        avg_ep_stat = LocalCallback.get_avg_ep_stat(ep_stat_buffer, prefix='eval/')
        avg_ep_stat['eval/eval_timeout'] = n_timeout

        duration = time.time() - t0
        avg_ep_stat['time/t_eval'] = duration
        avg_ep_stat['time/fps_eval'] = n_step * env.num_envs / duration

        for i in range(env.num_envs):
            env.set_attr('eval_mode', False, indices=i)
        obs = env.reset()
        return avg_ep_stat, ep_events

    @staticmethod
    def get_avg_ep_stat(ep_stat_buffer, prefix=''):
        avg_ep_stat = {}
        if len(ep_stat_buffer) > 0:
            for ep_info in ep_stat_buffer:
                for k, v in ep_info.items():
                    k_avg = f'{prefix}{k}'
                    if k_avg in avg_ep_stat:
                        avg_ep_stat[k_avg] += v
                    else:
                        avg_ep_stat[k_avg] = v

            n_episodes = float(len(ep_stat_buffer))
            for k in avg_ep_stat.keys():
                avg_ep_stat[k] /= n_episodes
            avg_ep_stat[f'{prefix}n_episodes'] = n_episodes

        return avg_ep_stat
