from __future__ import annotations

from dataclasses import dataclass
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class PPOConfig:
    # rollout / update
    rollout_steps: int = 2048
    update_epochs: int = 10
    minibatch_size: int = 64
    clip_range: float = 0.2
    # optimization
    lr: float = 5e-5
    max_grad_norm: float = 0.5

    # PPO loss
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5

    # logging / misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ActorCritic(nn.Module):

    def __init__(self, bev_shape, state_dim= 6, action_dim= 2):
        super().__init__()
        c, h, w = bev_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        # infer cnn output dim
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out = self.cnn(dummy)
            cnn_flat = int(out.view(1, -1).shape[1])

        self.cnn_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_flat, 256),
            nn.ReLU(inplace=True),
        )

        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(inplace=True),
        )

        self.actor_mean = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

        # log_std parameter (diagonal Gaussian). Initialized mildly.
        self.log_std = nn.Parameter(torch.ones(action_dim) * (-0.5))

    def forward(self, bev, state):
        x = self.cnn(bev)
        x = self.cnn_head(x)
        s = self.state_mlp(state)
        z = torch.cat([x, s], dim=1)
        z = self.fuse(z)
        mean = self.actor_mean(z)
        value = self.critic(z)
        return mean, self.log_std, value


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
        self.done = np.zeros((size,), dtype=np.float32)  # 1.0 if terminal at this step

        self.adv = np.zeros((size,), dtype=np.float32)
        self.ret = np.zeros((size,), dtype=np.float32)

        self.ptr = 0

    def add(self, obs, action, logp, value, reward, done):
        i = self.ptr
        self.bev[i] = obs["bev"]
        self.state[i] = obs["state"]
        self.action[i] = action
        self.logp[i] = logp
        self.value[i] = value
        self.reward[i] = reward
        self.done[i] = 1.0 if done else 0.0
        self.ptr += 1

    def compute_gae(self, last_value, gamma, lam):
        """
        GAE with episode boundaries using done flags.
        """
        adv = 0.0
        for t in reversed(range(self.size)):
            next_nonterminal = 1.0 - self.done[t]
            next_value = last_value if t == self.size - 1 else self.value[t + 1]
            delta = self.reward[t] + gamma * next_value * next_nonterminal - self.value[t]
            adv = delta + gamma * lam * next_nonterminal * adv
            self.adv[t] = adv
        self.ret = self.adv + self.value
        # normalize advantages
        self.adv = (self.adv - self.adv.mean()) / (self.adv.std() + 1e-8)

    def get_batches(self, minibatch_size: int, shuffle: bool = True):
        idx = np.arange(self.size)
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, self.size, minibatch_size):
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


class PPOAgent:
    def __init__(self, bev_shape, state_dim=6, action_dim=2, cfg=None):

        self.cfg = cfg if cfg is not None else PPOConfig()
        self.device = self.cfg.device

        self.net = ActorCritic(bev_shape, state_dim, action_dim).to(self.device)
        
        self.opt = optim.AdamW(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.lr_scheduler = None 

    def set_scheduler(self, total_updates):

        lr_lambda = lambda epoch: 1.0 - (epoch / float(total_updates))
        self.lr_scheduler = LambdaLR(self.opt, lr_lambda=lr_lambda)

    @torch.no_grad()
    def act(self, obs, deterministic= False):
        bev = torch.from_numpy(obs["bev"]).unsqueeze(0).to(self.device)  # (1,C,H,W)
        state = torch.from_numpy(obs["state"]).unsqueeze(0).to(self.device)  # (1,6)

        mean, log_std, value = self.net(bev, state)
        std = torch.exp(log_std)

        if deterministic:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()

        # clip to [-1, 1] since env expects it
        action_clipped = torch.clamp(action, -1.0, 1.0)

        # log prob under unclipped distribution but evaluated at clipped action (common practical choice)
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(action_clipped).sum(dim=-1)

        return action_clipped.squeeze(0).cpu().numpy().astype(np.float32), float(logp.item()), float(value.item())

    @torch.no_grad()
    def value(self, obs):
        bev = torch.from_numpy(obs["bev"]).unsqueeze(0).to(self.device)
        state = torch.from_numpy(obs["state"]).unsqueeze(0).to(self.device)
        _, _, v = self.net(bev, state)
        return float(v.item())

    def update(self, buffer):
        cfg = self.cfg
        losses_pi, losses_v, entropies, approx_kls = [], [], [], []

        for _ in range(cfg.update_epochs):
            for bev, state, action, old_logp, old_value, adv, ret in buffer.get_batches(cfg.minibatch_size):
                mean, log_std, value = self.net(bev, state)
                
                # 3. 수치 안정성: log_std를 안전한 범위로 클램핑 (-20 ~ 2)
                log_std = torch.clamp(log_std, -20, 2)
                std = torch.exp(log_std)

                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(action).sum(dim=-1)
                
                # Ratio 계산 시 수치 폭발 방지
                ratio = torch.exp(logp - old_logp)

                # PPO Loss 계산
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
                loss_pi = -torch.min(surr1, surr2).mean()

                value = value.squeeze(-1)
                v_loss = 0.5 * (value - ret).pow(2).mean()

                entropy = dist.entropy().sum(dim=-1).mean()
                loss = loss_pi + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                # 가중치 업데이트
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()

                # 통계 기록
                with torch.no_grad():
                    approx_kl = (old_logp - logp).mean().item()
                    approx_kls.append(approx_kl)

            # 4. KL Early Stopping (매우 중요): 
            # 한 에폭 내에서 정책이 너무 많이 변하면 업데이트 중단
            if abs(np.mean(approx_kls)) > 0.05: # 기준치: 0.01~0.05
                break

        # 5. 에폭 종료 후 스케줄러 스텝
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "loss_pi": np.mean(losses_pi) if losses_pi else 0.0,
            "loss_v": np.mean(losses_v) if losses_v else 0.0,
            "entropy": np.mean(entropies) if entropies else 0.0,
            "approx_kl": np.mean(approx_kls) if approx_kls else 0.0,
            "lr": self.opt.param_groups[0]['lr']
        }
    def save(self, path):
        ckpt = {
            "state_dict": self.net.state_dict(),
            "cfg": self.cfg.__dict__,
        }
        torch.save(ckpt, path)

    def load(self, path, map_location= None):
        ckpt = torch.load(path, map_location=map_location if map_location else self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        # cfg는 필요 시 사용자가 직접 맞추는 것을 권장(여기서는 load만)
        return ckpt
