"""Bounded PPO training and held-out evaluation for the offline BEV simulation."""
import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from local_logging import LocalRun, run_directory
from roach.ppo import PPOAgent, PPOConfig, RolloutBuffer
from task_reward import TaskEnv, TaskRewardConfig


def evaluate(env, agent, output, seeds, horizon):
    records = []
    for name in ('learned', 'fixed_original', 'random'):
        log = LocalRun(output / name)
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            rng = np.random.default_rng(seed + 10000)
            total_reward, speeds = 0., []
            collision = False
            terms_sum = {}
            started = time.time()
            for step in range(1, horizon + 1):
                if name == 'learned':
                    action = agent.act(obs, deterministic=True)[0]
                elif name == 'fixed_original':
                    inp = {k: torch.as_tensor(v).to(env.device).unsqueeze(0) for k, v in obs.items()}
                    action = env._policy.forward(inp, deterministic=True, clip_action=True)[0]
                else:
                    action = rng.uniform(-1, 1, size=2).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                task = info['task']
                speeds.append(task['speed_m_s'])
                total_reward += reward
                collision |= task['collision']
                for key,value in info['reward_terms'].items():
                    terms_sum[key] = terms_sum.get(key,0.) + value
                log.log({'seed': seed, 'episode_step': step, 'reward': reward,
                         'reward_terms': info['reward_terms'], 'task': task,
                         'terminated': terminated, 'truncated': truncated, 'reason': info['reason']}, step=step)
                if terminated or truncated:
                    break
            record = {'policy': name, 'seed': seed, 'steps': step,
                      'sim_seconds': step * env.FIXED_DT, 'wall_seconds': time.time()-started,
                      'return': total_reward, 'collision': collision,
                      'success': bool(info['success']), 'offroad': task['offroad'],
                      'route_completion': task['route_completion'], 'progress_m': task['route_progress_m'],
                      'mean_speed_m_s': float(np.mean(speeds)), 'final_cte_m': task['cte_m'],
                      'terminated': terminated, 'truncated': truncated, 'reason': info['reason'], 'offroute': info['reason']=='offroute',
                      'reward_terms_sum': terms_sum}
            records.append(record)
            print('EVAL',json.dumps(record),flush=True)
    (output / 'episodes.json').write_text(json.dumps(records,indent=2))
    summaries = {}
    for name in ('learned','fixed_original','random'):
        group = [r for r in records if r['policy'] == name]
        summaries[name] = {key: float(np.mean([r[key] for r in group])) for key in
                          ('return','collision','success','offroad','offroute','route_completion','progress_m','mean_speed_m_s','wall_seconds')}
    (output / 'summary.json').write_text(json.dumps(summaries,indent=2))
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description='PPO 학습 및 독립 seed 평가 (로컬 기록)')
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--rollout',type=int,default=32)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--town',default='Town03')
    parser.add_argument('--episode-steps',type=int)
    parser.add_argument('--eval-steps',type=int,default=32)
    parser.add_argument('--eval-seeds',type=int,nargs='+',default=[101,102,103])
    parser.add_argument('--skip-eval',action='store_true')
    parser.add_argument('--reward-config',type=Path)
    parser.add_argument('--checkpoint',type=Path,default=Path('roach/log/ckpt_11833344.pth'))
    parser.add_argument('--from-scratch',action='store_true')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    if args.steps < 2 or args.rollout < 2 or args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.eval_steps < 1:
        parser.error('steps/rollout >= 2, epochs/batch/eval_steps >= 1, lr > 0 are required')
    if not args.skip_eval and args.seed in args.eval_seeds:
        parser.error('Evaluation seeds must be independent of the training seed')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    reward_args = json.loads(args.reward_config.read_text()) if args.reward_config else {}
    if args.episode_steps is not None:
        reward_args['max_episode_steps'] = args.episode_steps
    reward_cfg = TaskRewardConfig(**reward_args)
    cfg = PPOConfig(rollout_steps=args.rollout,update_epochs=args.epochs,minibatch_size=args.batch_size,
                    lr=args.lr,device=args.device)
    out = args.output or run_directory(args.town)
    run = LocalRun(out, {**vars(args),'ppo':asdict(cfg),'reward':asdict(reward_cfg),'mode':'ppo'})
    versions = {name: importlib.metadata.version(name) for name in ('torch','numpy','gymnasium','pygame','omegaconf')}
    (out/'environment.json').write_text(json.dumps(versions,indent=2))
    env = TaskEnv(town=args.town,vis=False,render_mode='none',device=args.device,
                  FIXED_DT=1/30,sim_steps_per_frame=1,task_reward=reward_cfg)
    before = None
    started = time.time()
    try:
        obs,_ = env.reset(seed=args.seed)
        agent = PPOAgent(obs['birdview'].shape,obs['state'].shape[0],2,cfg)
        if not args.from_scratch:
            agent.load(str(args.checkpoint),strict=True)
        before = {k:v.detach().clone() for k,v in agent.net.state_dict().items()}
        buffer = RolloutBuffer(args.rollout,obs['birdview'].shape,6,2,args.device)
        global_step,updates,episode = 0,0,0
        episode_return = 0.
        while global_step < args.steps:
            buffer.reset()
            while buffer.ptr < args.rollout and global_step < args.steps:
                action, logp, value = agent.act(obs)
                next_obs,reward,terminated,truncated,info = env.step(action)
                next_value = 0. if terminated else agent.value(next_obs)
                if not np.isfinite([reward,logp,value,next_value]).all():
                    raise FloatingPointError('Non-finite rollout value/reward/logprob')
                buffer.add(obs,action,logp,value,reward,terminated or truncated,
                           terminated=terminated,truncated=truncated,next_value=next_value)
                global_step += 1
                episode_return += reward
                run.log({'episode':episode,'reward':reward,'reward_terms':info['reward_terms'],
                         'task':info['task'],'action':action,'logprob':logp,'value':value,
                         'next_value':next_value,'terminated':terminated,'truncated':truncated},step=global_step)
                obs = next_obs
                if terminated or truncated:
                    run.log({'episode/return':episode_return,'episode/reason':info['reason'],
                             'episode/steps':env.step_count},step=global_step)
                    episode += 1
                    episode_return = 0.
                    obs,_=env.reset(seed=args.seed + episode)
            with torch.no_grad():
                ratios = []
                for bev,state,actions,old_logp,*_ in buffer.get_batches(args.batch_size,shuffle=False):
                    distribution,_ = agent.net.get_dist_and_value(bev,state)
                    ratios.extend(torch.exp(distribution.log_prob(actions)-old_logp).cpu().tolist())
            error = float(np.max(np.abs(np.asarray(ratios)-1)))
            if not np.isfinite(error) or error > 1e-3:
                raise RuntimeError(f'Pre-update policy ratio error {error}, expected approximately 1')
            buffer.compute_gae(0.,cfg.gamma,cfg.gae_lambda)
            stats = agent.update(buffer)
            if not np.isfinite(list(stats.values())).all():
                raise FloatingPointError('Non-finite update metrics')
            updates += 1
            run.log({'update/index':updates,'preupdate_ratio_max_error':error,
                     **{'train/'+k:v for k,v in stats.items()}},step=global_step)
            print('UPDATE',updates,'STEP',global_step,'RATIO_ERROR',error,'STATS',stats,flush=True)
        changed = sum(not torch.equal(v,agent.net.state_dict()[k]) for k,v in before.items())
        if changed == 0:
            raise RuntimeError('No policy parameters changed')
        checkpoint = out/'ppo_final.pth'
        agent.save(str(checkpoint))
        restored = PPOAgent(obs['birdview'].shape,6,2,cfg)
        restored.load(str(checkpoint),strict=True,verbose=False)
        for k,v in agent.net.state_dict().items():
            torch.testing.assert_close(v,restored.net.state_dict()[k],atol=0,rtol=0)
        initial = agent.act(obs,deterministic=True)
        final = restored.act(obs,deterministic=True)
        np.testing.assert_array_equal(initial[0],final[0])
        assert initial[1:] == final[1:]
        summary = {'training_steps':global_step,'updates':updates,'changed_parameter_tensors':changed,
                   'training_wall_seconds':time.time()-started,'checkpoint':str(checkpoint),'reload_equal':True}
        if not args.skip_eval:
            env.task_reward = replace(reward_cfg,max_episode_steps=args.eval_steps)
            summary['evaluation'] = evaluate(env,restored,out/'evaluation',args.eval_seeds,args.eval_steps)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        print('FINISHED',json.dumps(summary),flush=True)
    finally:
        env.close()
        print('PPO simulator closed',flush=True)


if __name__ == '__main__':
    main()
