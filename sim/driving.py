import math
from .utility import clamp, to_rad_if_needed, normalize_angle_rad
import random
import numpy as np
import enum
from collections import deque
import carla

C0 = 0.10
C1 = 0.02
C2 = 0.003

INIT_SPEED        = 5.0
WHEELBASE         = 2.875
MAX_STEER_DEG     = 35.0

TURN_LOOKAHEAD_PTS = 20
V_TURN_MIN         = 6.0
K_CURV             = 20.0
STOP_DIST          = 15.0

A_MAX = 3.0
B_MAX = 6.0


class Gear(enum.IntEnum):
    P = 0   # Park
    R = 1   # Reverse
    N = 2   # Neutral
    D = 3   # Drive


class PID:
    def __init__(self, kp, ki, kd, out_min=-1.0, out_max=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.integral = 0.0
        self.prev_err = 0.0
        self.first    = True

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0
        self.first    = True

    def step(self, err, dt):
        if dt <= 1e-6:
            return 0.0
        derr         = 0.0 if self.first else (err - self.prev_err) / dt
        self.first    = False
        self.integral += err * dt
        u = self.kp * err + self.ki * self.integral + self.kd * derr
        u = clamp(u, self.out_min, self.out_max)
        if (u == self.out_max and err > 0) or (u == self.out_min and err < 0):
            self.integral *= 0.9
        self.prev_err = err
        return u

TOWN_Z_THRESHOLD = {
    "Town04": 4.0,
    "Town05": 3.0,
}
class BicycleModel:
    def __init__(self, x, y, z, yaw,id,town):
        self.x   = float(x)
        self.y   = float(y)
        self.z   = float(z)
        self.layer=-1
        if town=='Town04'or town=='Town05':
            if z>TOWN_Z_THRESHOLD[town]:
                self.layer=1


        self.id = int(id)
        self.steer   = 0
        self.throttle   = 0
        self.brake   = 0
        self.town=town
        self.yaw = to_rad_if_needed(yaw)

        self.v   = 0.0
        self.L   = float(WHEELBASE)
        self.max_steer = math.radians(MAX_STEER_DEG)
        self.gear      = Gear.D
        self.history   = deque(maxlen=16)
        for _ in range(16):
            self.history.append((self.x, self.y, self.z, self.yaw))

    def get_derivative(self, state, steer, throttle, brake):
        x, y, yaw, v = state

        a_drive      = 0.0
        actual_brake = brake * B_MAX

        if   self.gear == Gear.D: a_drive =  throttle * A_MAX
        elif self.gear == Gear.R: a_drive = -throttle * A_MAX
        elif self.gear == Gear.N: a_drive = 0.0
        elif self.gear == Gear.P: return np.array([0, 0, 0, -v * 10.0])

        a_drag  = np.sign(v) * (C0 + C1 * abs(v) + C2 * v ** 2) if abs(v) > 0.01 else 0.0
        a_brake = np.sign(v) * actual_brake                       if abs(v) > 0.01 else 0.0
        a       = a_drive - a_brake - a_drag

        if abs(v) < 0.05 and abs(a_drive) < 0.01:
            a = -v * 5.0

        dx   = v * math.cos(yaw)
        dy   = v * math.sin(yaw)
        dv   = a
        dyaw = (v / self.L) * math.tan(steer) if abs(v) > 0.05 else 0.0

        return np.array([dx, dy, dyaw, dv])

    def update(self, steer, throttle, brake, dt):
        if self.gear == Gear.P:
            self.v = 0.0
            return
        self.steer=steer
        self.throttle=throttle
        self.brake=brake

        state = np.array([self.x, self.y, self.yaw, self.v])
        k1 = self.get_derivative(state, steer, throttle, brake)
        k2 = self.get_derivative(state + k1 * dt / 2, steer, throttle, brake)
        k3 = self.get_derivative(state + k2 * dt / 2, steer, throttle, brake)
        k4 = self.get_derivative(state + k3 * dt,     steer, throttle, brake)

        new_state        = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.x, self.y, self.yaw, self.v = new_state
        self.yaw         = normalize_angle_rad(self.yaw)
        self.history.append((self.x, self.y, self.z, self.yaw))

    def get_transform(self):
        return carla.Transform(
            carla.Location(x=float(self.x), y=float(self.y), z=float(self.z)),
            carla.Rotation(pitch=0.0, yaw=math.degrees(self.yaw), roll=0.0)
        )

    def get_velocity(self):
        return carla.Vector3D(
            x=float(self.v * math.cos(self.yaw)),
            y=float(self.v * math.sin(self.yaw)),
            z=0.0
        )
    def use_dynamic_z(self) -> bool:
        """Town04, Town05 같은 특정 타운만 z 체크."""
        return self.town in TOWN_Z_THRESHOLD

    def get_z_threshold(self):
        return TOWN_Z_THRESHOLD.get(self.town, None)

    def has_crossed_z_threshold(self, prev_z: float, curr_z: float) -> bool:

        th = self.get_z_threshold()
        if th is None:
            return False

        return ((prev_z < th <= curr_z) or
                (prev_z > th >= curr_z))

    def update_z(self, new_z: float) -> bool:
        if not self.use_dynamic_z():
            self.z = float(new_z)
            return False

        prev_z = self.z
        curr_z = float(new_z)

        crossed = self.has_crossed_z_threshold(prev_z, curr_z)

        self.prev_z = prev_z
        self.z = curr_z
        if crossed:
            self.layer= self.layer * -1


# ---------------------------------------------------------------------------
# 경로 유틸
# ---------------------------------------------------------------------------
def find_lookahead_index(path, start_idx, x, y, lookahead_dist):
    n = len(path)
    for i in range(start_idx, n):
        dx = path[i][0] - x
        dy = path[i][1] - y
        if math.hypot(dx, dy) >= lookahead_dist:
            return i
    return n - 1


def pure_pursuit_control(vehicle, target_x, target_y, lookahead_dist):
    alpha = math.atan2(target_y - vehicle.y, target_x - vehicle.x) - vehicle.yaw
    alpha = normalize_angle_rad(alpha)
    return math.atan2(2.0 * vehicle.L * math.sin(alpha), lookahead_dist)


def estimate_curvature(path, idx, n_ahead=TURN_LOOKAHEAD_PTS):
    n  = len(path)
    i2 = min(n - 1, idx + n_ahead)
    i1 = min(n - 1, idx + max(2, n_ahead // 2))

    x0, y0 = path[idx][0], path[idx][1]
    x1, y1 = path[i1][0],  path[i1][1]
    x2, y2 = path[i2][0],  path[i2][1]

    a1 = math.atan2(y1 - y0, x1 - x0)
    a2 = math.atan2(y2 - y1, x2 - x1)
    da = abs(normalize_angle_rad(a2 - a1))
    d  = math.hypot(x2 - x0, y2 - y0) + 1e-6
    return da / d


def remaining_distance(path, idx):
    dist = 0.0
    for i in range(idx, len(path) - 1):
        dist += math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
    return dist


def find_closest_index(path, x, y):
    min_dist    = float('inf')
    closest_idx = 0
    for i in range(len(path)):
        dist = math.hypot(path[i][0] - x, path[i][1] - y)
        if dist < min_dist:
            min_dist    = dist
            closest_idx = i
    return closest_idx


# ---------------------------------------------------------------------------
# 공유 플래너 로직 (LongitudinalPlanner / NPCLongitudinalPlanner 공통)
# ---------------------------------------------------------------------------
def _follow_logic_common(veh, all_vehicles, all_ped, cruise_speed,
                          min_follow_dist, time_gap, lookahead_dist,
                          ped_corridor_half, ped_radius, a_comf):
    """
    전방 차량 추종 + 보행자 감속 공통 로직.
    최대 허용 속도를 반환한다.
    """
    v_follow = cruise_speed

    for veh_info in all_vehicles:
        other = veh_info["veh"]
        if other is veh:
            continue

        dx   = other.x - veh.x
        dy   = other.y - veh.y
        dist = math.hypot(dx, dy)

        if dist < lookahead_dist:
            forward = np.array([math.cos(veh.yaw), math.sin(veh.yaw)])
            target  = np.array([dx, dy]) / (dist + 1e-6)
            dot     = np.dot(forward, target)

            if dot > 0.92:
                safe_dist = min_follow_dist + veh.v * time_gap
                if dist < safe_dist:
                    follow_v_candidate = other.v * (
                        max(0, dist - min_follow_dist) /
                        (safe_dist - min_follow_dist + 1e-6)
                    )
                    v_follow = min(v_follow, follow_v_candidate)

    if all_ped is not None:
        forward = np.array([math.cos(veh.yaw), math.sin(veh.yaw)])
        best_s  = None

        for ped in all_ped:
            dx = ped.x - veh.x
            dy = ped.y - veh.y

            s = dx * forward[0] + dy * forward[1]
            if s <= 0.0 or s > lookahead_dist:
                continue

            l = abs(forward[0] * dy - forward[1] * dx)
            if l > ped_corridor_half:
                continue

            if best_s is None or s < best_s:
                best_s = s

        if best_s is not None:
            stop_offset      = WHEELBASE + ped_radius + 0.5
            target_stop_dist = max(0.0, best_s - stop_offset)
            if target_stop_dist < 25.0:
                v_ped_stop = math.sqrt(2.0 * a_comf * target_stop_dist)
                v_follow   = min(v_follow, v_ped_stop)

    return v_follow


def _apply_curvature_speed_common(path, idx, cruise_speed):
    """곡률 기반 속도 제한 (공통)."""
    curv    = estimate_curvature(path, idx)
    v_curve = cruise_speed / (1.0 + K_CURV * curv)
    return min(cruise_speed, max(V_TURN_MIN, v_curve))


# ---------------------------------------------------------------------------
# Ego 종방향 플래너
# ---------------------------------------------------------------------------
class LongitudinalPlanner:
    def __init__(self):
        self.active_tl_id      = -1
        self.is_waiting        = False
        self.min_follow_dist   = 6
        self.time_gap          = 1.2
        self.lookahead_dist    = 40.0
        self.traffic           = False
        self.ped_corridor_half = 2.0
        self.ped_radius        = 0.8
        self.a_comf            = 1.8
        self.dist_npc          = 40.0
        self.dist_to_stopline  = 20.0

    def compute_target_speed(self, path, closest_idx, ego_veh, all_vehicles, all_ped,
                              cruise_speed, tl_manager):
        base_v   = self._compute_base_speed(path, closest_idx, cruise_speed, tl_manager)
        follow_v = _follow_logic_common(
            ego_veh, all_vehicles, all_ped, cruise_speed,
            self.min_follow_dist, self.time_gap, self.lookahead_dist,
            self.ped_corridor_half, self.ped_radius, self.a_comf,
        )
        # dist_npc 업데이트 (정보 목적)
        self._update_dist_npc(ego_veh, all_vehicles)

        final_target_v = min(base_v, follow_v)

        if final_target_v < 0.2:
            self.is_waiting = True
            return 0.0

        self.is_waiting = False
        return final_target_v

    def _update_dist_npc(self, ego, all_vehicles):
        """가장 가까운 전방 NPC 거리를 기록 (디버그용)."""
        for veh_info in all_vehicles:
            other = veh_info["veh"]
            if other is ego:
                continue
            dx   = other.x - ego.x
            dy   = other.y - ego.y
            dist = math.hypot(dx, dy)
            if dist < self.lookahead_dist:
                forward = np.array([math.cos(ego.yaw), math.sin(ego.yaw)])
                target  = np.array([dx, dy]) / (dist + 1e-6)
                if np.dot(forward, target) > 0.92:
                    safe_dist = self.min_follow_dist + ego.v * self.time_gap
                    if dist < safe_dist:
                        self.dist_npc = dist

    def _compute_base_speed(self, path, closest_idx, cruise_speed, tl_manager):
        is_committed = True
        self.traffic = False
        lookback     = 5

        for i in range(max(0, closest_idx - lookback), closest_idx + 1):
            if path[i][7] == -1:
                is_committed = False
                break

        if is_committed:
            self.active_tl_id = -1
            return _apply_curvature_speed_common(path, closest_idx, cruise_speed)

        stop_idx = -1
        for i in range(closest_idx, min(len(path), closest_idx + 40)):
            if path[i][7] != -1:
                stop_idx = i
                break

        if stop_idx != -1:
            for k in range(closest_idx, min(len(path), stop_idx + 5)):
                if path[k][8] != -1:
                    self.active_tl_id = path[k][8]
                    break

        if self.active_tl_id != -1:
            st = tl_manager.state(self.active_tl_id)
            if st in ["R", "Y"]:
                self.dist_to_stopline = 0
                if stop_idx != -1:
                    for j in range(closest_idx, stop_idx):
                        self.dist_to_stopline += math.hypot(
                            path[j + 1][0] - path[j][0],
                            path[j + 1][1] - path[j][1]
                        )

                stop_offset      = WHEELBASE + 0.5
                target_stop_dist = max(0.0, self.dist_to_stopline - stop_offset)

                if target_stop_dist < 20.0:
                    self.traffic = True
                    v_stop       = math.sqrt(2.0 * 1.5 * target_stop_dist)
                    return min(cruise_speed, v_stop)

        return _apply_curvature_speed_common(path, closest_idx, cruise_speed)


# ---------------------------------------------------------------------------
# NPC 종방향 플래너
# ---------------------------------------------------------------------------
class NPCLongitudinalPlanner:
    def __init__(self):
        self.active_tl_id          = -1
        self.is_waiting            = False
        self.min_follow_dist       = 5
        self.time_gap              = 1.0
        self.stop_random_offset    = random.uniform(-0.5, 1.5)
        self.yellow_go_threshold   = random.uniform(1.0, 2.0)
        self.ped_corridor_half     = 2.0
        self.ped_radius            = 0.8
        self.a_comf                = 1.8
        self.lookahead_dist        = 20.0

    def compute_target_speed(self, path, closest_idx, veh, all_vehicles, all_ped,
                              cruise_speed, tl_manager):
        base_v   = self._compute_base_speed(path, closest_idx, veh.v, cruise_speed, tl_manager)
        follow_v = _follow_logic_common(
            veh, all_vehicles, all_ped, cruise_speed,
            self.min_follow_dist, self.time_gap, self.lookahead_dist,
            self.ped_corridor_half, self.ped_radius, self.a_comf,
        )
        final_v = min(base_v, follow_v)

        if final_v < 0.2:
            self.is_waiting = True
            return 0.0

        self.is_waiting = False
        return final_v

    def _compute_base_speed(self, path, closest_idx, current_v, cruise_speed, tl_manager):
        is_committed = True
        lookback     = 5

        for i in range(max(0, closest_idx - lookback), closest_idx + 1):
            if path[i][7] == -1:
                is_committed = False
                break

        if is_committed:
            self.active_tl_id = -1
            return _apply_curvature_speed_common(path, closest_idx, cruise_speed)

        stop_idx = -1
        for i in range(closest_idx, min(len(path), closest_idx + 40)):
            if path[i][7] != -1:
                stop_idx = i
                break

        if stop_idx != -1:
            for k in range(closest_idx, min(len(path), stop_idx + 5)):
                if path[k][8] != -1:
                    self.active_tl_id = path[k][8]
                    break

        if self.active_tl_id != -1:
            st               = tl_manager.state(self.active_tl_id)
            dist_to_stopline = 0.0
            if stop_idx != -1:
                for j in range(closest_idx, stop_idx):
                    dist_to_stopline += math.hypot(
                        path[j + 1][0] - path[j][0],
                        path[j + 1][1] - path[j][1]
                    )

            if st == "Y":
                ttz = dist_to_stopline / max(current_v, 0.1)
                if ttz < self.yellow_go_threshold:
                    return _apply_curvature_speed_common(path, closest_idx, cruise_speed)

            if st in ["R", "Y"]:
                stop_offset      = WHEELBASE + 0.5 + self.stop_random_offset
                target_stop_dist = max(0.0, dist_to_stopline - stop_offset)
                if target_stop_dist < 20.0:
                    v_stop = math.sqrt(2.0 * 1.3 * target_stop_dist)
                    return min(cruise_speed, v_stop)

        return _apply_curvature_speed_common(path, closest_idx, cruise_speed)
