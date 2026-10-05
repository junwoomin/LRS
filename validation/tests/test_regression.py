"""CPU regression of the supplied Roach checkpoint and the actual PPO rollout."""
import importlib.abc
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))


class NoExternalTracker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'wandb':
            raise ModuleNotFoundError('The external tracker is intentionally unavailable')


sys.meta_path.insert(0, NoExternalTracker())
import numpy as np
import torch
from task_reward import TaskEnv, TaskRewardConfig
from roach.ppo import PPOAgent, PPOConfig, RolloutBuffer
from roach.utils.checkpoint import load_checkpoint
from roach.models.ppo_policy import PpoPolicy
from roach.utils.local_callback import LocalCallback

torch.set_num_threads(2)
torch.manual_seed(0)
np.random.seed(0)
out = PROJECT / 'validation' / 'regression'
out.mkdir(parents=True, exist_ok=True)
env = TaskEnv(town='Town03', vis=False, render_mode='none', device='cpu', task_reward=TaskRewardConfig(max_episode_steps=2))
try:
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    agent = PPOAgent(obs['birdview'].shape, obs['state'].shape[0], 2,
                     PPOConfig(device='cpu', rollout_steps=4, update_epochs=1, minibatch_size=4))
    agent.load('roach/log/ckpt_11833344.pth', strict=True)
    original, _ = PpoPolicy.load('roach/log/ckpt_11833344.pth')
    inp = {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}
    with torch.no_grad():
        dist, value = agent.net.get_dist_and_value(inp['birdview'], inp['state'])
        features, _ = original._get_features(**inp)
        source_dist, alpha, beta = original._get_action_dist_from_features(features)
        source_value = original.value_head(features)
        torch.testing.assert_close(dist.alpha, torch.as_tensor(alpha), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(dist.beta, torch.as_tensor(beta), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(value, source_value, atol=1e-6, rtol=1e-5)
        source_action = original.unscale_action(source_dist.get_actions(deterministic=True).numpy())
        np.testing.assert_allclose(agent.act(obs, deterministic=True)[0], source_action[0], atol=1e-6)
    print('PASS: exact original head mapping, normalized features, Beta parameters, value and deterministic action')

    buffer = RolloutBuffer(4, obs['birdview'].shape, 6, 2, 'cpu')
    before = {k: v.clone() for k, v in agent.net.state_dict().items()}
    for _ in range(4):
        action, logp, value = agent.act(obs)
        nxt, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        assert np.isfinite(logp) and np.isfinite(value) and np.isfinite(reward)
        buffer.add(obs, action, logp, value, reward, done, terminated=terminated, truncated=truncated, next_value=0. if terminated else agent.value(nxt))
        obs = env.reset(seed=1)[0] if done else nxt
    ratios = []
    for bev, state, actions, old_logp, *_ in buffer.get_batches(4, shuffle=False):
        with torch.no_grad():
            distribution, _ = agent.net.get_dist_and_value(bev, state)
            ratio = torch.exp(distribution.log_prob(actions) - old_logp)
            torch.testing.assert_close(ratio, torch.ones_like(ratio), atol=1e-4, rtol=1e-4)
            ratios = ratio.tolist()
    print('PASS: pre-update ratio approximately 1', ratios)
    buffer.compute_gae(0.0 if done else agent.value(obs), .99, .9)
    stats = agent.update(buffer)
    assert all(np.isfinite(v) for v in stats.values()), stats
    changed = sum(not torch.equal(v, agent.net.state_dict()[k]) for k, v in before.items())
    assert changed > 0
    print('PASS: finite losses/KL/logprob and parameter update', stats, 'changed_tensors', changed)
    ckpt = out / 'reload_test.pth'
    agent.save(str(ckpt))
    reloaded = PPOAgent(obs['birdview'].shape, 6, 2, agent.cfg)
    reloaded.load(str(ckpt), strict=True)
    for k, value in agent.net.state_dict().items():
        torch.testing.assert_close(reloaded.net.state_dict()[k], value, atol=0, rtol=0)
    expected = agent.act(obs, deterministic=True)
    actual = reloaded.act(obs, deterministic=True)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]
    print('PASS: save/reload parameter and action/value/logprob equality')
    # Same tensors under another path do not receive the historical metadata allowance.
    import shutil, pickle
    untrusted_copy = out / 'legacy_copy.pth'
    shutil.copyfile('roach/log/ckpt_11833344.pth', untrusted_copy)
    try:
        load_checkpoint(untrusted_copy)
    except pickle.UnpicklingError:
        print('PASS: external/copied metadata checkpoint is rejected')
    else:
        raise AssertionError('Unexpected legacy metadata allowance outside the exact checkpoint path')
    finally:
        untrusted_copy.unlink()
    assert 'wandb' not in sys.modules
    print('PASS: imports, simulation and PPO regression completed with tracker import blocked')
    (out / 'result.json').write_text(json.dumps({'ratio': ratios, 'stats': stats, 'changed_tensors': changed}, indent=2))
finally:
    env.close()
