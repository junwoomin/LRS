import cv2
import math
import numpy as np
from shapely.geometry import Polygon
import pygame
import random
import heapq
import json
import carla
from collections import deque
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def normalize_angle_rad(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def to_rad_if_needed(a: float) -> float:
    a = float(a)
    if abs(a) > 2.0 * math.pi + 1e-6:
        return math.radians(a)
    return a

def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))


def _wrap_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def _cell_key(x, y, grid):
    return (int(math.floor(x / grid)), int(math.floor(y / grid)))

def world_to_pixel(x, y, OFFSET_X, OFFSET_Y, PPM):
    px = (x - OFFSET_X) * PPM
    py = (y - OFFSET_Y) * PPM
    return int(px), int(py)

def to_mask(img, thr=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray > thr



def overlay_solid(canvas_rgb, mask, color_rgb, alpha=1.0):
    out = canvas_rgb.copy()
    if alpha >= 1.0:
        out[mask] = color_rgb
        return out
    out_masked = out[mask].astype(np.float32)
    col = np.array(color_rgb, dtype=np.float32)
    out[mask] = (out_masked * (1 - alpha) + col * alpha).astype(np.uint8)
    return out


def apply_mask_color(img, mask, color, alpha=1.0):
    if not np.any(mask):
        return img
    color_arr = np.array(color, dtype=np.float32)
    img_f = img.astype(np.float32)
    img_f[mask] = img_f[mask] * (1.0 - alpha) + color_arr * alpha
    return np.clip(img_f, 0, 255).astype(np.uint8)


def blit_rgb_to_surface(dst_surf, rgb):
    pygame.surfarray.blit_array(dst_surf, rgb.swapaxes(0, 1))




# ---------------------------------------------------------------------------
# carla
# ---------------------------------------------------------------------------
def carla_rot_to_mat(carla_rotation):
    roll  = np.deg2rad(carla_rotation.roll)
    pitch = np.deg2rad(carla_rotation.pitch)
    yaw   = np.deg2rad(carla_rotation.yaw)

    yaw_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    pitch_matrix = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0,  np.cos(pitch)]
    ])
    roll_matrix = np.array([
        [1, 0, 0],
        [0,  np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)]
    ])
    return yaw_matrix.dot(pitch_matrix).dot(roll_matrix)


def vec_global_to_ref(target_vec_in_global, ref_rot_in_global):
    R = carla_rot_to_mat(ref_rot_in_global)
    np_vec = np.array([[target_vec_in_global.x],
                        [target_vec_in_global.y],
                        [target_vec_in_global.z]])
    np_vec_ref = R.T.dot(np_vec)
    return carla.Vector3D(x=np_vec_ref[0, 0], y=np_vec_ref[1, 0], z=np_vec_ref[2, 0])



def check_ref_vehicle_collision_from_bev(
    ego_local_mask: np.ndarray,
    vehicle_local_bev: np.ndarray,
    overlap_px_th: int = 1,
):
    ego_bin = ego_local_mask > 0
    veh_bin = vehicle_local_bev > 0

    overlap_px = int(np.count_nonzero(ego_bin & veh_bin))
    collided = overlap_px >= overlap_px_th
    return collided, overlap_px




def world_to_ego_bev(world_x, world_y, ego_x, ego_y, ego_yaw, bev_size, ppm):
    dx = world_x - ego_x
    dy = world_y - ego_y

    cos_theta = math.cos(-ego_yaw + math.pi / 2)
    sin_theta = math.sin(-ego_yaw + math.pi / 2)

    rel_x = dx * cos_theta - dy * sin_theta
    rel_y = dx * sin_theta + dy * cos_theta

    screen_x = (bev_size / 2) + (rel_x * ppm)
    screen_y = (bev_size / 2) - (rel_y * ppm)

    return int(screen_x), int(screen_y)


def global_pixel_to_local_bev(px, py, ego_cx, ego_cy, ego_yaw,
                               out_size=200, out_mpp=0.20, pixels_per_meter=5.0,
                               move_px=0):
    u0 = out_size / 2.0
    v0 = out_size / 2.0 + move_px

    k = out_mpp * pixels_per_meter

    c = math.cos(ego_yaw)
    s = math.sin(ego_yaw)

    dx = px - ego_cx
    dy = py - ego_cy

    u = u0 + (-s * dx + c * dy) / k
    v = v0 + (-c * dx - s * dy) / k

    return int(round(u)), int(round(v))





# ---------------------------------------------------------------------------
# BEV 렌더링
# ---------------------------------------------------------------------------


def make_input_vis_rgb(das, path, lane,
                       c_vehicle_history,
                       c_walker_history,
                       c_tl_history):
    H, W = das.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)

    vis = apply_mask_color(vis, das > 0,  (60, 60, 60),    alpha=1.0)
    vis = apply_mask_color(vis, path > 0, (0, 180, 255),   alpha=1.0)
    vis = apply_mask_color(vis, lane > 0, (255, 255, 255),  alpha=1.0)

    n = len(c_vehicle_history)

    for i, veh in enumerate(c_vehicle_history):
        alpha = 1.0 - (i / max(n - 1, 1)) * 0.7
        vis = apply_mask_color(vis, veh > 0, (0, 0, 255), alpha=alpha)

    for i, ped in enumerate(c_walker_history):
        alpha = 1.0 - (i / max(n - 1, 1)) * 0.7
        vis = apply_mask_color(vis, ped > 0, (0, 255, 0), alpha=alpha)

    for i, tl in enumerate(c_tl_history):
        alpha = 1.0 - (i / max(n - 1, 1)) * 0.7
        vis = apply_mask_color(vis, tl == 80,  (0, 255, 0),   alpha=alpha)
        vis = apply_mask_color(vis, tl == 170, (255, 255, 0), alpha=alpha)
        vis = apply_mask_color(vis, tl == 255, (255, 0, 0),   alpha=alpha)

    return vis


# ---------------------------------------------------------------------------
# 경로 유틸
# ---------------------------------------------------------------------------
def get_distance_to_next_junction(path, current_idx):
    total_dist = 0.0
    if path[current_idx][7] != -1:
        return 0.0
    for i in range(current_idx, len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        total_dist += d
        if p2[7] != -1:
            return total_dist
    return float('inf')


# ---------------------------------------------------------------------------
# 보행자 그래프 / 에이전트
# ---------------------------------------------------------------------------
class PedGraph:
    def __init__(self, json_path: str, grid=2.0):
        payload = json.load(open(json_path, "r", encoding="utf-8"))
        self.town   = payload.get("town", "Unknown")
        self.nodes  = payload["nodes"]
        self.edges  = payload["edges"]
        self.spawns = payload.get("spawn_points", [])

        self.xyz = [(float(n["x"]), float(n["y"]), float(n["z"])) for n in self.nodes]
        self.N   = len(self.nodes)

        self.adj = [[] for _ in range(self.N)]
        for e in self.edges:
            u = int(e["u"]); v = int(e["v"]); w = float(e["w"])
            t = e.get("type", "")
            self.adj[u].append((v, w, t))
            self.adj[v].append((u, w, t))

        self.grid    = float(grid)
        self.spatial = {}
        for i, (x, y, z) in enumerate(self.xyz):
            k = _cell_key(x, y, self.grid)
            self.spatial.setdefault(k, []).append(i)

    def nearest_node(self, x, y):
        qk = _cell_key(x, y, self.grid)
        best  = None
        bestd = float("inf")
        for R in range(0, 30):
            found = False
            for dx in range(-R, R + 1):
                for dy in range(-R, R + 1):
                    k = (qk[0] + dx, qk[1] + dy)
                    if k not in self.spatial:
                        continue
                    found = True
                    for idx in self.spatial[k]:
                        px, py, pz = self.xyz[idx]
                        d = (x - px) ** 2 + (y - py) ** 2
                        if d < bestd:
                            bestd = d
                            best  = idx
            if found and best is not None:
                return best
        return best

    def spawn_to_node(self, spawn_idx: int):
        sp = self.spawns[int(spawn_idx)]
        return self.nearest_node(float(sp["x"]), float(sp["y"]))

    def astar(self, start: int, goal: int, jaywalk_cost=1.0):
        gx, gy, gz = self.xyz[goal]

        def h(n):
            x, y, z = self.xyz[n]
            return math.hypot(x - gx, y - gy)

        pq     = [(h(start), 0.0, start)]
        best_g = {start: 0.0}
        parent = {start: -1}

        while pq:
            f, g, cur = heapq.heappop(pq)
            if cur == goal:
                path = [cur]
                while parent[path[-1]] != -1:
                    path.append(parent[path[-1]])
                path.reverse()
                return path

            if g > best_g.get(cur, float("inf")) + 1e-9:
                continue

            for nxt, w, etype in self.adj[cur]:
                mult = jaywalk_cost if etype == "jaywalk" else 1.0
                ng   = g + w * mult
                if ng < best_g.get(nxt, float("inf")):
                    best_g[nxt]  = ng
                    parent[nxt]  = cur
                    heapq.heappush(pq, (ng + h(nxt), ng, nxt))

        return None


class PedAgent:
    def __init__(self, graph: PedGraph, speed=1.4, reach=0.6, max_yaw_rate=6.0, jaywalk_cost=1.0):
        self.g            = graph
        self.v            = float(speed)
        self.reach        = float(reach)
        self.max_yaw_rate = float(max_yaw_rate)
        self.jaywalk_cost = float(jaywalk_cost)

        self.x   = 0.0
        self.y   = 0.0
        self.z   = 0.0
        self.yaw = 0.0
        self.path_nodes = []
        self.path_xy    = []
        self.i     = 0
        self.alive = True
        self.history = deque(maxlen=16)
        for _ in range(16):
            self.history.append((self.x, self.y, self.z, self.yaw))

        ok = self.reset_random_route()
        if not ok:
            self.alive = False

    def reset_random_route(self, max_tries=5):
        for _ in range(max_tries):
            if self.g.spawns:
                s_idx = random.randrange(len(self.g.spawns))
                start = self.g.spawn_to_node(s_idx)
            else:
                start = random.randrange(self.g.N)

            if start is None:
                continue

            goal = random.randrange(self.g.N)
            while goal == start:
                goal = random.randrange(self.g.N)

            path = self.g.astar(start, goal, jaywalk_cost=self.jaywalk_cost)
            if not path or len(path) < 2:
                continue

            self.path_nodes = path
            self.path_xy    = [self.g.xyz[n] for n in path]
            self.i          = 0
            self.x, self.y, self.z = self.path_xy[0]

            nx, ny, nz = self.path_xy[1]
            self.yaw = math.atan2(ny - self.y, nx - self.x)
            return True

        return False

    def step(self, dt: float):
        if not self.alive:
            return
        if not self.path_xy or self.i >= len(self.path_xy) - 1:
            self.reset_random_route()
            return

        tx, ty, tz = self.path_xy[self.i + 1]
        dx = tx - self.x
        dy = ty - self.y
        d  = math.hypot(dx, dy)

        if d <= self.reach:
            self.i += 1
            if self.i >= len(self.path_xy) - 1:
                self.reset_random_route()
            return

        target_yaw = math.atan2(dy, dx)
        yaw_err    = _wrap_pi(target_yaw - self.yaw)
        yaw_step   = max(-self.max_yaw_rate * dt, min(self.max_yaw_rate * dt, yaw_err))
        self.yaw   = _wrap_pi(self.yaw + yaw_step)

        self.x += math.cos(self.yaw) * self.v * dt
        self.y += math.sin(self.yaw) * self.v * dt
        self.z  = tz
        self.history.append((self.x, self.y, self.z, self.yaw))
