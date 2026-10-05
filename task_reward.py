"""Opt-in task reward and proper episode boundaries for the existing offline simulator."""
from dataclasses import dataclass, asdict
import math
import random
import numpy as np
from loop import BEVPathFollowEnv


@dataclass
class TaskRewardConfig:
    progress_per_m: float = 1.0
    time_per_s: float = 0.05
    cte_per_s: float = 0.5
    heading_per_s: float = 0.2
    overspeed_per_s: float = 0.5
    reverse_per_s: float = 0.5
    idle_per_s: float = 0.2
    control_change: float = 0.01
    collision_cost: float = 20.0
    offroute_cost: float = 10.0
    red_light_cost: float = 10.0
    stalled_cost: float = 5.0
    success_bonus: float = 10.0
    max_cte_m: float = 2.5
    max_speed_m_s: float = 40 / 3.6
    idle_speed_m_s: float = 0.1
    stall_seconds: float = 5.0
    success_distance_m: float = 1.5
    success_cte_m: float = 0.75
    allowed_stop_distance_m: float = 5.0
    max_episode_steps: int = 2000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and non-negative')
        for name in ('max_cte_m', 'max_speed_m_s', 'stall_seconds', 'max_episode_steps'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')


def reward_terms(signals, cfg):
    """Distance rewards, costs per simulation second, exclusive fatal-event costs."""
    failure = ('collision' if signals['collision'] else
               'red_light' if signals['red_light'] else
               'offroute' if signals['offroad'] or abs(signals['cte_m']) > cfg.max_cte_m else
               'stalled' if signals['idle_seconds'] >= cfg.stall_seconds else None)
    if failure:
        # No progress/success bonus or second failure penalty on an unsafe step.
        return {failure: -getattr(cfg, failure + '_cost')}, True, failure
    dt = signals['dt_s']
    lane_gate = max(0.0, 1 - abs(signals['cte_m']) / cfg.max_cte_m)
    heading_gate = max(0.0, math.cos(signals['heading_rad']))
    terms = {
        'progress': cfg.progress_per_m * max(0., signals['new_progress_m']) * lane_gate * heading_gate,
        'time': -cfg.time_per_s * dt,
        'cte': -cfg.cte_per_s * min(4., (signals['cte_m'] / cfg.max_cte_m) ** 2) * dt,
        'heading': -cfg.heading_per_s * (signals['heading_rad'] / math.pi) ** 2 * dt,
        'overspeed': -cfg.overspeed_per_s * max(0., signals['speed_m_s'] - cfg.max_speed_m_s) / cfg.max_speed_m_s * dt,
        'reverse': -cfg.reverse_per_s * max(0., -signals['speed_m_s']) * dt,
        'idle': -cfg.idle_per_s * dt if signals['idle'] and not signals['allowed_stop'] else 0.,
        'control_change': -cfg.control_change * signals['action_change_squared'],
    }
    if signals['success']:
        terms['success'] = cfg.success_bonus
    return terms, bool(signals['success']), 'success' if signals['success'] else None


def credited_progress(route_s, frontier_s, travelled_m, dt_s, cfg):
    delta = max(0., route_s - frontier_s)
    paid = min(delta, travelled_m * 1.25, cfg.max_speed_m_s * dt_s)
    return paid, max(frontier_s, route_s)


def project_route(points, arc, position, previous_segment):
    # A local window prevents jumps to distant portions of a crossing/looping route.
    lo = max(0, previous_segment - 3)
    hi = min(len(points) - 1, previous_segment + 13)
    starts = points[lo:hi]
    vectors = points[lo+1:hi+1] - starts
    squared = np.sum(vectors * vectors, axis=1)
    t = np.clip(np.sum((position-starts)*vectors, axis=1) / np.maximum(squared, 1e-12), 0., 1.)
    nearest = starts + vectors * t[:, None]
    distance = np.linalg.norm(position-nearest, axis=1)
    i = int(np.argmin(distance))
    segment = lo + i
    return (float(arc[segment] + t[i] * math.sqrt(squared[i])), segment,
            float(distance[i]), math.atan2(vectors[i, 1], vectors[i, 0]))


def crosses_segment(a, b, c, d):
    # Strict crossings avoid a stationary vehicle on a line being charged repeatedly.
    def cross(u, v):
        return float(u[0]*v[1]-u[1]*v[0])
    return cross(d-c,a-c)*cross(d-c,b-c) < 0 and cross(b-a,c-a)*cross(b-a,d-a) <= 0


class TaskEnv(BEVPathFollowEnv):
    """Five-tuple Gymnasium interface; base simulation/observations remain intact."""
    def __init__(self, *args, task_reward=None, **kwargs):
        self.task_reward = task_reward or TaskRewardConfig()
        self._episode_ended = False
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
        if seed is not None:
            # Route generation uses Python random and some simulation code uses NumPy global RNG.
            random.seed(seed)
            np.random.seed(seed)
        obs, info = super().reset(seed=seed, options=options)
        self._route_points = np.asarray([p[:2] for p in self.ego_path], dtype=np.float64)
        length = np.linalg.norm(np.diff(self._route_points, axis=0), axis=1)
        self._route_arc = np.concatenate([[0.], np.cumsum(length)])
        self._task_position = np.array([self.ego_vehicle.x, self.ego_vehicle.y])
        self._route_s, self._route_segment, _, _ = project_route(self._route_points, self._route_arc, self._task_position, 0)
        self._frontier_s = self._route_s
        self._initial_s = self._route_s
        self._idle_seconds = 0.
        self._last_action = np.zeros(2)
        self._episode_ended = False
        self.total_reward = 0.
        return obs, info

    def step(self, action):
        if self._episode_ended:
            raise RuntimeError('reset() is required after an episode ends')
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 2 or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError('PPO action must contain two finite values in [-1, 1]')
        obs, old_reward, _, info = super().step(action)
        pos = np.array([self.ego_vehicle.x, self.ego_vehicle.y])
        route_s, segment, cte, route_heading = project_route(self._route_points, self._route_arc, pos, self._route_segment)
        travelled = float(np.linalg.norm(pos - self._task_position))
        new_progress, self._frontier_s = credited_progress(route_s, self._frontier_s, travelled, self.FIXED_DT, self.task_reward)
        heading = (route_heading - self.ego_vehicle.yaw + math.pi) % (2*math.pi) - math.pi
        collision = bool(self.veh_collision)
        offroad = bool(self.das_collision)
        red_light = False
        red_wait = False
        light_id = int(self.ego_path[segment][8])
        light_state = self.tl_manager.state(light_id) if light_id != -1 else None
        stopline = self.tl_stoplines.get(light_id)
        if stopline is not None and light_state == 'R':
            left = np.array([stopline['left'].x, stopline['left'].y])
            right = np.array([stopline['right'].x, stopline['right'].y])
            red_light = crosses_segment(self._task_position, pos, left, right)
            center = (left + right) / 2
            forward = np.array([math.cos(route_heading), math.sin(route_heading)])
            longitudinal = float(np.dot(center-pos, forward))
            lateral = abs(float(np.cross(forward, center-pos)))
            red_wait = (0 <= longitudinal <= self.task_reward.allowed_stop_distance_m and
                        lateral <= np.linalg.norm(right-left)/2 + .5)
        forward = np.array([math.cos(self.ego_vehicle.yaw), math.sin(self.ego_vehicle.yaw)])
        npc_wait = False
        nearest_front_m = None
        for npc in self.npc_vehicle_agents:
            if not npc.get('alive', True):
                continue
            other = npc['veh']
            if abs(other.z - self.ego_vehicle.z) > 2.:
                continue
            relative = np.array([other.x, other.y])-pos
            longitudinal = float(np.dot(relative, forward))
            lateral = abs(float(np.cross(forward, relative)))
            if longitudinal > 0 and lateral < 2.:
                nearest_front_m = longitudinal if nearest_front_m is None else min(nearest_front_m,longitudinal)
                npc_wait |= longitudinal < self.task_reward.allowed_stop_distance_m
        speed = float(self.ego_vehicle.v)
        idle = abs(speed) < self.task_reward.idle_speed_m_s and new_progress < 1e-3
        allowed_stop = red_wait or npc_wait
        self._idle_seconds = self._idle_seconds + self.FIXED_DT if idle and not allowed_stop else 0.
        success = (self._route_arc[-1] - route_s <= self.task_reward.success_distance_m and
                   cte <= self.task_reward.success_cte_m and abs(heading) < math.pi / 4)
        signals = {'dt_s': self.FIXED_DT, 'new_progress_m': new_progress,
                   'route_progress_m': max(0., self._frontier_s-self._initial_s),
                   'route_length_m': float(self._route_arc[-1]),
                   'route_completion': float(np.clip(self._frontier_s / max(self._route_arc[-1],1e-9),0,1)),
                   'speed_m_s': speed, 'cte_m': cte, 'heading_rad': heading,
                   'collision': collision, 'offroad': offroad, 'red_light': red_light,
                   'red_wait': red_wait, 'npc_wait': npc_wait, 'nearest_front_m': nearest_front_m,
                   'allowed_stop': allowed_stop, 'idle': idle, 'idle_seconds': self._idle_seconds,
                   'success': success,
                   'action_change_squared': float(np.sum((action-self._last_action)**2))}
        terms, terminated, reason = reward_terms(signals, self.task_reward)
        truncated = not terminated and self.step_count >= self.task_reward.max_episode_steps
        self._episode_ended = terminated or truncated
        self.reason = reason or ('time_limit' if truncated else None)
        reward = float(sum(terms.values()))
        self.total_reward += reward - old_reward
        info.update({'task': signals, 'reward_terms': terms, 'collision': collision,
                     'success': bool(reason == 'success'), 'terminated': terminated,
                     'truncated': truncated, 'reason': self.reason})
        self._route_s, self._route_segment = route_s, segment
        self._task_position = pos
        self._last_action = action.copy()
        return obs, reward, terminated, truncated, info
