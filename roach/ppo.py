from __future__ import annotations
from roach.utils.checkpoint import load_checkpoint

from dataclasses import dataclass, asdict
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RoachXtMaCNN(nn.Module):
    def __init__(self, bev_shape, state_dim, features_dim=256, states_neurons=(256, 256)):
        super().__init__()
        c, h, w = bev_shape

        self.features_dim = features_dim

        # checkpoint 인덱스와 맞추기 위해 ReLU를 Sequential 안에 같이 넣음
        self.cnn = nn.Sequential(
            nn.Conv2d(c,   8,  kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(8,   16, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(16,  32, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(32,  64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(64,  128, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1), nn.ReLU(),
        )

        state_layers = []
        prev = state_dim
        for hidden in states_neurons:
            state_layers.append(nn.Linear(prev, hidden))
            state_layers.append(nn.ReLU())
            prev = hidden
        self.state_linear = nn.Sequential(*state_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            x = self.cnn(dummy)
            n_flatten = x.flatten(start_dim=1).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + prev, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, bev, state):
        x = self.cnn(bev)
        x = x.flatten(start_dim=1)

        s = self.state_linear(state)
        x = torch.cat([x, s], dim=1)
        x = self.linear(x)
        return x

class BetaActionDistribution:
    """
    action range [-1, 1] 을 위한 Beta 분포 wrapper
    내부적으로는 [0,1] Beta를 쓰고, 외부 action만 [-1,1]로 변환
    """
    def __init__(self, alpha, beta, eps=1e-6):
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.base_dist = torch.distributions.Beta(alpha, beta)

    def sample(self):
        u = self.base_dist.sample().clamp(self.eps, 1 - self.eps)  # bounded samples
        a = 2.0 * u - 1.0                    # [-1, 1]
        return a

    def deterministic_action(self):
        # Match the supplied Roach Beta policy mode, including boundary cases.
        alpha, beta = self.alpha, self.beta
        u = torch.zeros_like(alpha)
        u[:, 1] = 0.5
        interior = (alpha > 1) & (beta > 1)
        u[interior] = (alpha[interior] - 1) / (alpha[interior] + beta[interior] - 2)
        u[(alpha <= 1) & (beta > 1)] = 0
        u[(alpha > 1) & (beta <= 1)] = 1
        both_small = (alpha <= 1) & (beta <= 1)
        u[both_small] = self.base_dist.mean[both_small]
        return 2 * u - 1

    def log_prob(self, action):
        # [-1,1] -> [0,1]
        u = ((action + 1.0) * 0.5).clamp(self.eps, 1.0 - self.eps)
        # change-of-variables: u = (a+1)/2  => du/da = 1/2
        logp = self.base_dist.log_prob(u).sum(dim=-1) - action.shape[-1] * np.log(2.0)
        return logp

    def entropy(self):
        return self.base_dist.entropy().sum(dim=-1) + self.alpha.shape[-1] * np.log(2.0)
class RolloutBuffer:
    def __init__(self, size, bev_shape, state_dim, action_dim, device):
        self.size = size
        self.device = device

        c, h, w = bev_shape
        self.bev = np.zeros((size, c, h, w), dtype=np.float32)
        self.state = np.zeros((size, state_dim), dtype=np.float32)
        self.action = np.zeros((size, action_dim), dtype=np.float32)
        self.logp = np.zeros((size,), dtype=np.float32)
        self.value = np.zeros((size,), dtype=np.float32)
        self.reward = np.zeros((size,), dtype=np.float32)
        self.done = np.zeros((size,), dtype=np.float32)
        self.terminated = np.zeros((size,), dtype=np.float32)
        self.next_value = np.full((size,), np.nan, dtype=np.float32)

        self.adv = np.zeros((size,), dtype=np.float32)
        self.ret = np.zeros((size,), dtype=np.float32)

        self.ptr = 0

    def add(self, obs, action, logp, value, reward, done, terminated=None, truncated=False, next_value=None):
        if self.ptr >= self.size:
            raise IndexError("Rollout buffer is full")
        i = self.ptr
        self.terminated[i] = float(done if terminated is None else terminated)
        self.next_value[i] = np.nan if next_value is None else next_value
        self.bev[i] = obs["birdview"]
        self.state[i] = obs["state"]
        self.action[i] = action
        self.logp[i] = logp
        self.value[i] = value
        self.reward[i] = reward
        self.done[i] = 1.0 if done else 0.0
        self.ptr += 1

    def compute_gae(self, last_value, gamma, lam):
        n = self.ptr
        adv = 0.0
        for t in reversed(range(n)):
            trace_nonterminal = 1.0 - self.done[t]
            bootstrap_nonterminal = 1.0 - self.terminated[t]
            next_value = self.next_value[t]
            if np.isnan(next_value):
                next_value = last_value if t == n - 1 else self.value[t + 1]
            delta = self.reward[t] + gamma * next_value * bootstrap_nonterminal - self.value[t]
            adv = delta + gamma * lam * trace_nonterminal * adv
            self.adv[t] = adv

        self.ret[:n] = self.adv[:n] + self.value[:n]

        adv_slice = self.adv[:n]
        self.adv[:n] = (adv_slice - adv_slice.mean()) / (adv_slice.std() + 1e-8)

    def get_batches(self, minibatch_size: int, shuffle: bool = True):
        idx = np.arange(self.ptr)
        if shuffle:
            np.random.shuffle(idx)

        for start in range(0, self.ptr, minibatch_size):
            mb = idx[start:start + minibatch_size]
            yield self._to_tensors(mb)

    def _to_tensors(self, mb_idx: np.ndarray):
        bev = torch.from_numpy(self.bev[mb_idx]).to(self.device)
        state = torch.from_numpy(self.state[mb_idx]).to(self.device)
        action = torch.from_numpy(self.action[mb_idx]).to(self.device)
        old_logp = torch.from_numpy(self.logp[mb_idx]).to(self.device)
        old_value = torch.from_numpy(self.value[mb_idx]).to(self.device)
        adv = torch.from_numpy(self.adv[mb_idx]).to(self.device)
        ret = torch.from_numpy(self.ret[mb_idx]).to(self.device)
        return bev, state, action, old_logp, old_value, adv, ret

    def reset(self):
        self.ptr = 0
        self.next_value.fill(np.nan)

@dataclass
class PPOConfig:
    rollout_steps: int = 2048
    update_epochs: int = 10
    minibatch_size: int = 64
    clip_range: float = 0.2

    lr: float = 5e-5
    max_grad_norm: float = 0.5

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    target_kl: float = 0.03


class ActorCritic(nn.Module):
    """
    '내 스타일'의 단순한 하나의 net 객체를 유지하면서,
    내부 구조는 최대한 ROACH 스타일로 맞춘 버전.
    """
    def __init__(self, bev_shape, state_dim=5, action_dim=2):
        super().__init__()

        self.features_extractor = RoachXtMaCNN(
            bev_shape=bev_shape,
            state_dim=state_dim,
            features_dim=256,
            states_neurons=(256, 256),
        )

        # config의 policy_head_arch / value_head_arch = [256, 256]
        self.policy_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )

        self.mu_head = nn.Linear(256, action_dim)
        self.sigma_head = nn.Linear(256, action_dim)

        self.value_out = nn.Linear(256, 1)

    def forward(self, bev, state):
        feat = self.features_extractor(bev.float() / 255.0, state)

        pi_feat = self.policy_head(feat)
        vf_feat = self.value_head(feat)

        # Match the original checkpoint: Softplus without an added offset.
        mu = F.softplus(self.mu_head(pi_feat)).clamp_min(torch.finfo(pi_feat.dtype).tiny)
        sigma = F.softplus(self.sigma_head(pi_feat)).clamp_min(torch.finfo(pi_feat.dtype).tiny)

        value = self.value_out(vf_feat)
        return mu, sigma, value

    def get_dist_and_value(self, bev, state):
        mu, sigma, value = self.forward(bev, state)
        dist = BetaActionDistribution(mu, sigma)
        return dist, value
    


class PPOAgent:
    def __init__(self, bev_shape, state_dim=5, action_dim=2, cfg=None):
        self.cfg = cfg if cfg is not None else PPOConfig()
        self.device = self.cfg.device
        if torch.device(self.device).type == "cuda":
            # Keep single-observation collection and batched logprob recomputation in FP32.
            torch.backends.cudnn.allow_tf32 = False

        self.bev_shape = bev_shape
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = ActorCritic(bev_shape, state_dim, action_dim).to(self.device)

        self.opt = optim.AdamW(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.lr_scheduler = None

    def set_scheduler(self, total_updates):
        lr_lambda = lambda epoch: 1.0 - (epoch / float(total_updates))
        self.lr_scheduler = LambdaLR(self.opt, lr_lambda=lr_lambda)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        bev = torch.from_numpy(obs["birdview"]).float().unsqueeze(0).to(self.device)
        state = torch.from_numpy(obs["state"]).float().unsqueeze(0).to(self.device)

        dist, value = self.net.get_dist_and_value(bev, state)

        if deterministic:
            action = dist.deterministic_action()
        else:
            action = dist.sample()

        logp = dist.log_prob(action)

        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            float(logp.item()),
            float(value.item()),
        )

    @torch.no_grad()
    def value(self, obs):
        bev = torch.from_numpy(obs["birdview"]).float().unsqueeze(0).to(self.device)
        state = torch.from_numpy(obs["state"]).float().unsqueeze(0).to(self.device)
        _, value = self.net.get_dist_and_value(bev, state)
        return float(value.item())

    def update(self, buffer):
        cfg = self.cfg
        if cfg.minibatch_size < 1 or cfg.update_epochs < 1 or buffer.ptr < 1:
            raise ValueError("PPO needs positive epochs, batch size and a nonempty buffer")
        losses_pi, losses_v, entropies, approx_kls = [], [], [], []

        for _ in range(cfg.update_epochs):
            for bev, state, action, old_logp, old_value, adv, ret in buffer.get_batches(cfg.minibatch_size):
                dist, value = self.net.get_dist_and_value(bev, state)

                logp = dist.log_prob(action)
                log_ratio = logp - old_logp
                ratio = torch.exp(log_ratio)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                if not np.isfinite(approx_kl):
                    raise FloatingPointError("PPO KL became non-finite")
                if approx_kl > cfg.target_kl:
                    break

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
                loss_pi = -torch.min(surr1, surr2).mean()

                value = value.squeeze(-1)
                loss_v = 0.5 * (value - ret).pow(2).mean()

                entropy = dist.entropy().mean()

                loss = loss_pi + cfg.vf_coef * loss_v - cfg.ent_coef * entropy

                if not torch.isfinite(loss):
                    raise FloatingPointError("PPO loss became non-finite")
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                self.opt.step()

                losses_pi.append(loss_pi.item())
                losses_v.append(loss_v.item())
                entropies.append(entropy.item())
                approx_kls.append(approx_kl)

            if approx_kl > cfg.target_kl:
                break

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "loss_pi": np.mean(losses_pi) if losses_pi else 0.0,
            "loss_v": np.mean(losses_v) if losses_v else 0.0,
            "entropy": np.mean(entropies) if entropies else 0.0,
            "approx_kl": np.mean(approx_kls) if approx_kls else 0.0,
            "lr": self.opt.param_groups[0]["lr"],
        }

    def save(self, path):
        ckpt = {
            # ROACH save 형식도 같이 맞춤
            "policy_state_dict": self.net.state_dict(),
            "policy_init_kwargs": {
                "bev_shape": self.bev_shape,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "features_dim": 256,
                "states_neurons": [256, 256],
                "policy_head_arch": [256, 256],
                "value_head_arch": [256, 256],
                "distribution": "beta",
            },
            "train_init_kwargs": {
                "cfg": asdict(self.cfg),
            },
        }
        torch.save(ckpt, path)

    def load(self, path, map_location=None, strict=False, verbose=True):
        ckpt = load_checkpoint(path, map_location=map_location if map_location else self.device)

        if isinstance(ckpt, dict):
            if "policy_state_dict" in ckpt:
                state_dict = ckpt["policy_state_dict"]   # ROACH 형식
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]          # 네 예전 형식
            else:
                state_dict = ckpt                        # raw state_dict
        else:
            state_dict = ckpt

        # Verified Roach heads: Linear+Softplus distribution heads and final Linear value layer.
        head_names = {"dist_mu.0.weight": "mu_head.weight", "dist_mu.0.bias": "mu_head.bias",
                      "dist_sigma.0.weight": "sigma_head.weight", "dist_sigma.0.bias": "sigma_head.bias",
                      "value_head.4.weight": "value_out.weight", "value_head.4.bias": "value_out.bias"}
        expected = self.net.state_dict()
        cleaned = {}
        for k, v in state_dict.items():
            nk = k
            for prefix in ("net.", "policy.", "module."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            nk = head_names.get(nk, nk)
            if nk in expected and v.shape != expected[nk].shape:
                raise ValueError(f"Checkpoint shape mismatch for {nk}: {v.shape} != {expected[nk].shape}")
            cleaned[nk] = v

        missing, unexpected = self.net.load_state_dict(cleaned, strict=False)

        if verbose:
            print(f"[load] total loaded keys: {len(cleaned)}")
            if missing:
                print("[load] missing keys:")
                for x in missing:
                    print("   ", x)
            if unexpected:
                print("[load] unexpected keys:")
                for x in unexpected:
                    print("   ", x)

        if strict and (len(missing) > 0 or len(unexpected) > 0):
            raise RuntimeError(
                f"State dict mismatch.\nMissing: {missing}\nUnexpected: {unexpected}"
            )

        return ckpt






