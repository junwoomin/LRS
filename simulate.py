"""Run the supplied fixed Roach policy in the offline simulator without learning."""
import argparse
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import cv2
from loop import BEVPathFollowEnv
from local_logging import LocalRun, run_directory


def main(argv=None):
    parser = argparse.ArgumentParser(description='고정 Roach 정책의 오프라인 시뮬레이션 (학습 없음)')
    parser.add_argument('--steps', type=int, default=int(os.getenv('TOTAL_TIMESTEPS', '600')))
    parser.add_argument('--town', default=os.getenv('TOWN', 'Town03'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--headless', action='store_true', default=os.getenv('VIS', '1') == '0')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--legacy-gui', action='store_true', help='기존 지도 뷰어 사용')
    parser.add_argument('--autostart', action='store_true')
    parser.add_argument('--snapshot', action='store_true', help='실제 Pygame 창을 마지막 프레임 PNG로 기록')
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error('--steps must be positive')
    if not args.headless and not args.legacy_gui:
        from gui import main as gui_main
        options=['--mode','simulate','--steps',str(args.steps),'--town',args.town,'--seed',str(args.seed)]
        if args.output is not None:options+=['--output',str(args.output)]
        if args.autostart:options+=['--autostart']
        if args.snapshot:options+=['--snapshot']
        return gui_main(options)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(int(os.getenv('OMP_NUM_THREADS', '2')))
    output = args.output or run_directory(args.town)
    run = LocalRun(output, {**vars(args), 'mode': 'fixed_policy_simulation'})
    env = BEVPathFollowEnv(town=args.town, vis=not args.headless, device=args.device,
                          render_mode='none' if args.headless else 'human',
                          sim_steps_per_frame=1, FIXED_DT=1 / 30)
    writer = None
    started = time.time()
    completed = 0
    total_reward = 0.0
    try:
        obs, _ = env.reset(seed=args.seed)
        initial = (env.ego_vehicle.x, env.ego_vehicle.y)
        if not env.observation_space.contains(obs):
            raise RuntimeError('Observation does not match declared space')
        writer = cv2.VideoWriter(str(output / 'simulation.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (400, 400))
        if not writer.isOpened():
            raise RuntimeError('Cannot open simulation video writer')
        for step in range(1, args.steps + 1):
            inp = {k: torch.as_tensor(v).to(env.device).unsqueeze(0) for k, v in obs.items()}
            action, *_ = env._policy.forward(inp, deterministic=True, clip_action=True)
            obs, reward, done, info = env.step(action)
            actual_collision = bool(env.veh_collision)
            offroad = bool(env.das_collision)
            info.update({'collision': actual_collision, 'offroad': offroad,
                         'reward': reward, 'action': action,
                         'x': env.ego_vehicle.x, 'y': env.ego_vehicle.y})
            run.log(info, step=step)
            total_reward += reward
            completed = step
            frame = cv2.resize(env.input_vis_rgb, (400, 400), interpolation=cv2.INTER_NEAREST)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not args.headless:
                env.render()
            if done:
                break
        if args.snapshot and env._screen is not None:
            import pygame
            pygame.image.save(env._screen, str(output / 'gui.png'))
        summary = {'mode': 'fixed_policy_simulation', 'steps': completed,
                   'reward': total_reward, 'wall_seconds': time.time() - started,
                   'position_change_m': float(np.hypot(env.ego_vehicle.x-initial[0], env.ego_vehicle.y-initial[1])),
                   'speed_m_s': float(env.ego_vehicle.v), 'collision': actual_collision,
                   'offroad': offroad, 'output': str(output)}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        if writer is not None:
            writer.release()
        env.close()
        print('Simulator and video writer closed')


if __name__ == '__main__':
    main()
