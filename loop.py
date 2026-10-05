import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import deque

import numpy as np
import cv2

from sim.traffic import *
from sim.utility import *
from sim.routeplanner import *
from sim.driving import *
from sim.Global_Dynamic_MapManager import *

import gymnasium as gym
from gymnasium import spaces

from omegaconf import OmegaConf
from roach.models.ppo_policy import PpoPolicy

from omegaconf import OmegaConf
import torch as th

import torch
import time

# ======================================================================
# 상수
# ======================================================================
TL_VALUE_MAP = {
    "G": np.uint8(80),
    "Y": np.uint8(170),
    "R": np.uint8(255),
}


# ======================================================================
# 보상 설정
# ======================================================================
@dataclass
class RewardConfig:
    min_speed_kmh: float         = 10.0
    target_speed_kmh: float      = 30.0
    max_speed_kmh: float         = 40.0
    max_distance: float          = 2.5
    max_angle_center_lane: float = 20.0
    penalty_reward: float        = -10.0
    safety_distance: float       = 15.0
    w_stationary: float          = 0.2
    max_stationary_steps: int    = 30
    progress_weight: float       = 0.9
    center_weight: float         = 1.4
    heading_weight: float        = 1.0
    speed_weight: float          = 0.5
    smooth_weight: float         = 0.12
    cte_growth_weight: float     = 0.40
    center_bonus_cte_m: float    = 0.35
    center_bonus_heading_deg: float = 6.0
    center_bonus: float          = 0.30
    steer_residual_scale: float  = 0.30
    speed_residual_scale_ms: float = 1.8
    action_smoothing: float      = 0.35
    guidance_blend: float        = 0.80
    safety_blend: float          = 0.65


# ======================================================================
# 데이터 경로 헬퍼
# ======================================================================
def resolve_data_dir(town: str, ppm: float) -> str:

    return f"sim_using_data/data{int(ppm)}/{town}"


# ======================================================================
# 뷰어 상태 (zoom / pan)
# ======================================================================
@dataclass
class ViewerState:
    """렌더링 뷰어의 확대/이동 상태."""
    zoom: float   = 2.0       # 배율 (1.0 = 1:1, 2.0 = 2× 확대)
    center_x: float = 0.0     # 맵 픽셀 좌표 (현재 화면 중심)
    center_y: float = 0.0
    follow_ego: bool = True   # True이면 매 프레임 ego 차량 추적

    zoom_min: float = 0.2
    zoom_max: float = 10.0
    zoom_step: float = 1.05   # 한 프레임당 확대/축소 비율
    pan_speed: float = 8.0    # 한 프레임당 이동 픽셀 (맵 좌표 기준)


# ======================================================================
# 환경
# ======================================================================
class BEVPathFollowEnv(gym.Env):

    metadata = {"render_modes": ["human", "none"]}

    # ------------------------------------------------------------------
    # 생성자
    # ------------------------------------------------------------------
    def __init__(
        self,
        town                = "Town01",
        vis                 = False,
        render_mode         = "none",
        out_size            = 192,
        panel_update_every  = 3,
        ppm                 = 5.0,        # ★ 통합 PPM (기존 20/5 하드코딩 제거)
        render_fps          = 60,
        sim_steps_per_frame = 3,
        num_routes          = 20,
        waypoint_sample_dist = 1000.0,
        cruise_speed        = 20.0,
        loop_route          = False,
        end_on_route_end    = True,
        max_steps           = 2000,
        reward_cfg          = None,
        seed                = None,
        logger              = None,
        FIXED_DT            = 0.01,
        device              = None,
        initial_zoom        = 2.0,        # ★ 뷰어 초기 확대 배율
    ):
        super().__init__()
        self.logger = logger
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.town        = town
        self.vis         = bool(vis)
        self.render_mode = render_mode if render_mode in ("human", "none") else "none"

        self.RENDER_FPS          = int(render_fps)
        self.SIM_STEPS_PER_FRAME = int(sim_steps_per_frame)
        self.FIXED_DT            = FIXED_DT
        self.move_px             = 40

        # ── 통합 PPM ─────────────────────────────────────────────────
        self.PPM = float(ppm)

        self.OUT_SIZE           = int(out_size)
        self.PANEL_UPDATE_EVERY = int(panel_update_every)

        self.NUM_ROUTES            = int(num_routes)
        self.WAYPOINT_SAMPLE_DIST  = float(waypoint_sample_dist)
        self.LOOP_ROUTE            = bool(loop_route)
        self.END_ON_ROUTE_END      = bool(end_on_route_end)

        self.max_steps  = int(max_steps)
        self.reward_cfg = reward_cfg if reward_cfg is not None else RewardConfig()

        self._np_random = np.random.RandomState(seed)

        # ── 액션 / 관측 ─────────────────────────────────────────────
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.curr_target_speed_kmh      = float(self.reward_cfg.target_speed_kmh)
        self.curr_min_speed_kmh         = float(self.reward_cfg.min_speed_kmh)
        self.curr_max_distance          = float(self.reward_cfg.max_distance)
        self.curr_max_angle_center_lane = float(self.reward_cfg.max_angle_center_lane)

        self.roadoption_to_idx = {-1: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}

        self.observation_space = spaces.Dict({
            "birdview": spaces.Box(low=0.0, high=255.0,
                                shape=(15, self.OUT_SIZE, self.OUT_SIZE),
                                dtype=np.float32),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(6,),
                                dtype=np.float32),
        })

        # ── 차량 스프라이트 ──────────────────────────────────────────
        if not hasattr(self, "car_base"):
            self.car_base = pygame.Surface(
                (int(4.69 * self.PPM), int(2.0 * self.PPM)), pygame.SRCALPHA
            )
            self.car_base.fill((255, 50, 50))
            self.npc_base = pygame.Surface(
                (int(4.69 * self.PPM), int(2.0 * self.PPM)), pygame.SRCALPHA
            )
            self.npc_base.fill((50, 150, 255))

        # ── 정적 자산 로드 ───────────────────────────────────────────
        self._load_static_assets()

        # ── 렌더러 상태 (lazy init) ──────────────────────────────────
        self._pygame_inited    = False
        self._screen           = None
        self._clock            = None
        self._hud_font         = None
        self._full_map_surface = None

        # ── 뷰어 (zoom / pan) ────────────────────────────────────────
        self._viewer = ViewerState(zoom=float(initial_zoom))

        self._reset_episode_state()

    # ==================================================================
    # 초기화 / 로딩
    # ==================================================================
    def _load_static_assets(self) -> None:
        self.current_lookahead = 5.0

        # ── 정책 네트워크 로드 ────────────────────────────────────────
        cfgs = OmegaConf.load("roach/config/config_agent.yaml")
        cfgs = OmegaConf.to_container(cfgs)

        _train_cfg = cfgs["training"]
        _ckpt = "roach/log/ckpt_11833344.pth"

        self._policy, _train_cfg["kwargs"] = PpoPolicy.load(_ckpt, device=self.device)
        self._policy = self._policy.eval().to(self.device)

        # ── 데이터 경로 (PPM 기반 자동 해석) ─────────────────────────
        town     = self.town
        data_dir = resolve_data_dir(town, self.PPM)

        self.map_data_dir   = data_dir
        self.das_mask_path  = f"{data_dir}/das_full.png"
        self.lane_mask_path = f"{data_dir}/lane_full.png"
        self.das_mask_path2  = f"{data_dir}/das_full_high.png"
        self.lane_mask_path2 = f"{data_dir}/lane_full_high.png"
        self.stop_mask_path = f"{data_dir}/stoplines.png"
        self.offset_path    = f"{data_dir}/world_offset.npy"
        self.xodr_path      = f"sim_using_data/Town/{town}.xodr"

        height_dir           = f"sim_using_data/height_estimator/{town}"
        self.height_low_path  = f"{height_dir}/height_low.png"
        self.height_high_path = f"{height_dir}/height_high.png"

        # ── 월드 오프셋 ──────────────────────────────────────────────
        if not os.path.exists(self.offset_path):
            raise FileNotFoundError(f"Offset file not found: {self.offset_path}")
        world_offset = np.load(self.offset_path)
        self.OFFSET_X, self.OFFSET_Y = float(world_offset[0]), float(world_offset[1])

        # ── 보행자 그래프 ────────────────────────────────────────────
        ped_json       = f"sim_using_data/Town/{town}_ped_graph.json"
        self.ped_graph = PedGraph(ped_json, grid=2.0)

        # ── 글로벌 마스크 로드 (그레이스케일 1회만) ────────────────────
        self.global_das_mask  = cv2.imread(self.das_mask_path,  cv2.IMREAD_GRAYSCALE)
        self.global_lane_mask = cv2.imread(self.lane_mask_path, cv2.IMREAD_GRAYSCALE)

        if self.global_das_mask  is None: raise FileNotFoundError(f"Missing: {self.das_mask_path}")
        if self.global_lane_mask is None: raise FileNotFoundError(f"Missing: {self.lane_mask_path}")

        if town in ("Town04", "Town05"):
            self.global_high_das_mask  = cv2.imread(self.das_mask_path2,  cv2.IMREAD_GRAYSCALE)
            self.global_high_lane_mask = cv2.imread(self.lane_mask_path2, cv2.IMREAD_GRAYSCALE)
        else:
            self.global_high_das_mask  = np.zeros_like(self.global_das_mask)
            self.global_high_lane_mask = np.zeros_like(self.global_lane_mask)

        self.das_merged  = cv2.add(self.global_high_das_mask,  self.global_das_mask)
        lane_merged = cv2.add(self.global_high_lane_mask, self.global_lane_mask)

        self.MAP_H, self.MAP_W = self.das_merged.shape[:2]
        self.map_h = self.MAP_H
        self.map_w = self.MAP_W

        canvas = np.zeros((self.MAP_H, self.MAP_W, 3), dtype=np.uint8)
        canvas[self.das_merged  > 10] = (90,  90,  90)
        canvas[lane_merged > 10] = (255, 255, 255)
        self.global_canvas = canvas

        self.height_low = cv2.imread(self.height_low_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(self.height_low_path) else None
        self.height_high = cv2.imread(self.height_high_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(self.height_high_path) else None
        if self.town in ("Town04", "Town05") and (self.height_low is None or self.height_high is None):
            raise FileNotFoundError(f"Required height maps missing for {self.town}")

        # ── 신호등 마스크 (★ 단채널 uint8) ───────────────────────────
        self.current_tl_mask = np.zeros((self.map_h, self.map_w), dtype=np.uint8)
        self.tl_mask_history = deque(maxlen=16)

        # ── 동적 맵 매니저 ───────────────────────────────────────────
        self.dynamic_map_manager = GlobalDynamicMapManager(
            global_map_shape=(self.map_h, self.map_w), town=town,
            maxlen=16,
        )

        # ── OpenDRIVE 맵 ────────────────────────────────────────────
        loader = OfflineTLLoader(
            data_dir=f"sim_using_data/Town",
            Town=town,
            t_green=10.0,
            t_yellow=3.0,
            t_all_red=2.0,
        )

        self.world_map    = loader.world_map
        self.tl_stoplines = loader.tl_stoplines
        self.tl_manager   = loader.tl_manager

        self.tl_patch_cache = self._build_tl_patch_cache(thickness=6)

        self.current_tl_snapshot = self.tl_manager.capture_snapshot()
        self._render_full_snapshot_into_mask(
            self.current_tl_snapshot, self.current_tl_mask
        )
        for _ in range(16):
            self.tl_mask_history.append(self.current_tl_mask.copy())

    # ------------------------------------------------------------------
    # 렌더러 lazy init
    # ------------------------------------------------------------------
    def _init_renderer_if_needed(self) -> None:
        if not self.vis or self._pygame_inited:
            return

        os.environ.setdefault("SDL_VIDEODRIVER", "x11")
        pygame.init()
        self._pygame_inited = True

        self.SCREEN_RES = 1000
        self.PANEL_W    = 200

        self._hud_font = pygame.font.SysFont("consolas", 18)
        self._screen   = pygame.display.set_mode(
            (self.SCREEN_RES + self.PANEL_W, self.SCREEN_RES)
        )
        pygame.display.set_caption(f"BEVPathFollowEnv - {self.town}")
        self._clock = pygame.time.Clock()

        self._full_map_surface = pygame.surfarray.make_surface(
            self.global_canvas.swapaxes(0, 1)
        )
        self._bev_vis_surfs = {
            "input": pygame.Surface((self.OUT_SIZE, self.OUT_SIZE))
        }

    # ==================================================================
    # 에피소드 상태
    # ==================================================================
    def _reset_episode_state(self) -> None:
        self.ego_path    = []
        self.ego_path_px = None
        self.path_idx    = 0

        self.ego_vehicle        = None
        self.npc_vehicle_agents = []
        self.ped_agents         = []
        self.cte_history        = []

        self.planner = LongitudinalPlanner()

        loader = OfflineTLLoader(
            data_dir=f"sim_using_data/Town",
            Town=self.town,
            t_green=10.0,
            t_yellow=3.0,
            t_all_red=2.0,
        )

        self.world_map    = loader.world_map
        self.tl_stoplines = loader.tl_stoplines
        self.tl_manager   = loader.tl_manager

        self.tl_patch_cache = self._build_tl_patch_cache(thickness=6)

        self.current_tl_snapshot = self.tl_manager.capture_snapshot()
        self._render_full_snapshot_into_mask(
            self.current_tl_snapshot, self.current_tl_mask
        )
        for _ in range(16):
            self.tl_mask_history.append(self.current_tl_mask.copy())
        self.throttle = 0.0
        self.brake    = 0.0
        self.steer    = 0.0

        self.step_count       = 0
        self.prev_closest_idx = 0
        self.prev_path_idx    = 0
        self.last_steer       = 0.0
        self.steer_cmd        = 0
        self.throttle_cmd     = 0
        self.brake_cmd        = 0

        self._last_target_v = 0.0
        self._last_cte      = 0.0

        self.bev_surfs = {
            "input": np.zeros((self.OUT_SIZE, self.OUT_SIZE, 15), dtype=np.uint8),
        }

    # ==================================================================
    # Gym API
    # ==================================================================
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random = np.random.RandomState(seed)

        self._reset_episode_state()
        self.total_reward  = 0
        self.wt            = 0
        self.tl_stop_t     = 0.0
        self.last_steer    = 0.0
        self.prev_action_0 = 0
        self.prev_action_1 = 0
        self.curr_action_0 = 0
        self.curr_action_1 = 0

        # ── ego 경로 생성 ────────────────────────────────────────────
        self.ego_path = []
        for _ in range(20):
            ego_results = generate_random_routes(
                self.world_map, self.das_merged,
                self.OFFSET_X, self.OFFSET_Y,
                num_routes=1, sample_dist=4000.0, max_len=12000,
                ppm=self.PPM, invert_y=False
            )
            candidate = ego_results[0] if ego_results else []
            if len(candidate) >= 80:
                self.ego_path = candidate
                break

        if len(self.ego_path) < 80:
            raise RuntimeError("Valid ego route not found")

        # ── NPC 경로 ─────────────────────────────────────────────────
        NUM_NPC    = 30
        npc_routes = generate_random_routes(
            self.world_map, self.das_merged,
            self.OFFSET_X, self.OFFSET_Y,
            num_routes=NUM_NPC, sample_dist=4000.0, max_len=8000,
            ppm=self.PPM, invert_y=False
        )

        # ── 보행자 ───────────────────────────────────────────────────
        self.ped_agents = []
        for _ in range(0):
            p = PedAgent(
                self.ped_graph,
                speed=np.random.uniform(1.3, 2.4),
                reach=0.7, max_yaw_rate=8.0, jaywalk_cost=0.7,
            )
            if p.alive:
                self.ped_agents.append(p)

        self.ego_path_px = np.array(
            [world_to_pixel(x, y, self.OFFSET_X, self.OFFSET_Y, self.PPM)
             for (x, y, _, _, _, _, _, _, _, _) in self.ego_path],
            dtype=np.int32,
        )

        # ── ego 차량 생성 ────────────────────────────────────────────
        self.ego_vehicle = BicycleModel(
            self.ego_path[0][0], self.ego_path[0][1],
            self.ego_path[0][2], self.ego_path[0][3],
            1, self.town,
        )
        self.ego_vehicle.v = float(INIT_SPEED) if "INIT_SPEED" in globals() else 0.0

        # ── NPC 차량 생성 ────────────────────────────────────────────
        self.npc_vehicle_agents = []
        ego_x, ego_y   = self.ego_vehicle.x, self.ego_vehicle.y
        MIN_SPAWN_DIST = 6.0

        for k, route in enumerate(npc_routes, start=2):
            if len(route) < 2:
                continue

            npc_start_x, npc_start_y = route[0][0], route[0][1]
            if math.hypot(npc_start_x - ego_x, npc_start_y - ego_y) < MIN_SPAWN_DIST:
                continue

            too_close = any(
                math.hypot(npc_start_x - other["veh"].x,
                           npc_start_y - other["veh"].y) < MIN_SPAWN_DIST
                for other in self.npc_vehicle_agents
            )
            if too_close:
                continue

            npc_vehicle   = BicycleModel(
                npc_start_x, npc_start_y, route[0][2], route[0][3], k, self.town
            )
            npc_vehicle.v = float(INIT_SPEED) if "INIT_SPEED" in globals() else 0.0

            self.npc_vehicle_agents.append({
                "path":       route,
                "veh":        npc_vehicle,
                "idx":        0,
                "planner":    NPCLongitudinalPlanner(),
                "alive":      True,
                "last_steer": 0.0,
            })

        closest_idx           = find_closest_index(
            self.ego_path, self.ego_vehicle.x, self.ego_vehicle.y
        )
        self.prev_closest_idx = closest_idx
        self.prev_path_idx    = self.path_idx

        self.all_vehicle_agents = [{"veh": self.ego_vehicle, "alive": True}]
        self.all_vehicle_agents += [
            n for n in self.npc_vehicle_agents if n.get("alive", True)
        ]

        self._update_bev_surfs()

        if self.vis and self.render_mode == "human":
            self._init_renderer_if_needed()

        val = self.ego_path[0][9].value
        idx = self.roadoption_to_idx[val]
        self.road_op      = np.zeros(7, dtype=np.float32)
        self.road_op[idx] = 1.0

        obs  = self._get_obs()
        info = self._get_info(
            success=False, collision=False, terminated=False, truncated=False
        )
        self.stationary_steps = 0
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action):
        terminated = False
        truncated  = False

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 2 or not np.all(np.isfinite(action)):
            raise ValueError("Action must contain two finite values: acceleration and steering")
        acc, steer = action
        if acc >= 0.0:
            throttle, brake = acc, 0.0
        else:
            throttle, brake = 0.0, np.abs(acc)
        self.throttle = np.clip(throttle, 0, 1)
        self.brake    = np.clip(brake,    0, 1)
        self.steer    = np.clip(steer,   -1, 1)

        self.collision_detected     = False
        self.DAS_collision_detected = False
        self.success                = False

        self.step_traffic_lights(self.FIXED_DT)
        self.ego_vehicle.update(self.steer, self.throttle, self.brake, self.FIXED_DT)
        self._update_npcs(self.FIXED_DT)

        self.step_update_dynamic_maps()

        self.path_idx = find_lookahead_index(
            self.ego_path, self.path_idx,
            self.ego_vehicle.x, self.ego_vehicle.y,
            self.current_lookahead,
        )
        if self.town == 'Town04' or self.town == 'Town05':
            hx, hy = world_to_pixel(
                self.ego_vehicle.x, self.ego_vehicle.y,
                self.OFFSET_X, self.OFFSET_Y, self.PPM,
            )
            if self.ego_vehicle.layer > 0:
                pixel = float(self.height_high[hy, hx])
            else:
                pixel = float(self.height_low[hy, hx])
            self.ego_vehicle.update_z((pixel / 255.0) * 16)

        self.veh_collision, self.das_collision = self._update_bev_surfs()
        reward, terminated, truncated = self.get_reward()
        self.step_count      += 1
        self.prev_closest_idx = self.closest_idx
        self.total_reward    += float(reward)

        obs  = self._get_obs()
        info = self._get_info(
            success=self.success, collision=self.collision_detected,
            terminated=terminated, truncated=truncated,
        )
        return obs, float(reward), (terminated or truncated), info

    # ==================================================================
    # 보상
    # ==================================================================
    def get_reward(self):
        rc         = self.reward_cfg
        reward     = 0.0
        terminated = False
        truncated  = False
        self.reason = None

        env_speed_ms     = float(self.ego_vehicle.v)
        self.closest_idx = find_closest_index(
            self.ego_path, self.ego_vehicle.x, self.ego_vehicle.y
        )
        signed_cte, heading_error, _ = self._compute_tracking_errors(self.closest_idx)
        cte_rate = signed_cte - float(self._last_cte)

        progress_idx    = min(max(0, self.closest_idx - self.prev_closest_idx), 2)
        lane_gate       = max(0.0, 1.0 - abs(signed_cte) / max(1e-6, self.curr_max_distance))
        progress_reward = float(progress_idx) * lane_gate

        center_reward  = math.exp(-abs(signed_cte)    / 0.75)
        heading_reward = math.exp(-abs(heading_error) / np.deg2rad(10.0))

        cte_growth_penalty = 0.0
        if abs(signed_cte) > abs(self._last_cte):
            cte_growth_penalty = rc.cte_growth_weight * min(1.0, abs(cte_rate) / 0.5)

        reward += rc.progress_weight * progress_reward
        reward += rc.center_weight   * center_reward
        reward += rc.heading_weight  * heading_reward
        reward -= cte_growth_penalty

        if (abs(signed_cte) < rc.center_bonus_cte_m and
                abs(heading_error) < np.deg2rad(rc.center_bonus_heading_deg)):
            reward += rc.center_bonus

        self._last_cte            = float(signed_cte)
        self._last_heading_error  = float(heading_error)
        self._last_cte_rate       = float(cte_rate)
        self._last_progress_ratio = float(
            self.closest_idx / max(1, len(self.ego_path) - 1)
        )

        if self.reason is not None and self.logger is not None:
            self.logger.info(f"[Step {self.step_count}] {self.reason}")

        return reward, terminated, truncated

    # ==================================================================
    # 관측
    # ==================================================================
    def _get_obs(self):
        bev          = np.stack(self.bev_surfs["input"], axis=0).astype(np.float32)
        ev_transform = self.ego_vehicle.get_transform()
        vel_w        = self.ego_vehicle.get_velocity()
        vel_ev       = vec_global_to_ref(vel_w, ev_transform.rotation)
        vel_xy       = np.array([vel_ev.x, vel_ev.y], dtype=np.float32)

        state = np.array([
            self.throttle,
            self.steer,
            self.brake,
            self.ego_vehicle.gear.value,
            vel_xy[0],
            vel_xy[1],
        ], dtype=np.float32)

        return {"birdview": bev, "state": state.astype(np.float32)}

    # ==================================================================
    # NPC 업데이트
    # ==================================================================

    def _update_npcs(self, dt):
        collision_detected = False

        for ped_npc in self.ped_agents:
            ped_npc.step(dt)

        npc_list   = []
        state_list = []

        for npc in self.npc_vehicle_agents:
            if not npc.get("alive", True):
                continue

            veh  = npc["veh"]
            path = npc["path"]
            idx  = int(npc["idx"])

            idx = find_lookahead_index(path, idx, veh.x, veh.y, self.current_lookahead)

            if idx >= len(path) - int(self.current_lookahead + 2):
                if self.LOOP_ROUTE:
                    start_point = carla.Location(x=float(veh.x), y=float(veh.y), z=0.0)
                    new_routes  = generate_random_routes_roop(
                        self.world_map, start_point,
                        num_routes=1, sample_dist=self.WAYPOINT_SAMPLE_DIST, max_len=4000,
                    )
                    npc["path"] = new_routes[0] if new_routes else path
                    path        = npc["path"]
                    idx         = 0
                    veh.x, veh.y, veh.yaw = path[0][0], path[0][1], path[0][3]
                    veh.v = float(INIT_SPEED) if "INIT_SPEED" in globals() else 0.0
                else:
                    npc["alive"] = False
                    continue

            npc["idx"] = idx

            veh_collision, das_collision, masks = self._update_bev_surfs(veh)
            if veh_collision or das_collision:
                npc["alive"] = False
            else:
                npc_list.append(np.stack(masks, axis=0).astype(np.float32))

                ev_transform = veh.get_transform()
                vel_w        = veh.get_velocity()
                vel_ev       = vec_global_to_ref(vel_w, ev_transform.rotation)
                vel_xy       = np.array([vel_ev.x, vel_ev.y], dtype=np.float32)

                state = np.array([
                    veh.throttle, veh.steer, veh.brake,
                    veh.gear.value, vel_xy[0], vel_xy[1],
                ], dtype=np.float32)
                state_list.append(state)

        if len(npc_list) != 0:
            all_npc_bev   = np.stack(npc_list,   axis=0)
            all_npc_state = np.stack(state_list, axis=0)

            npc_input = {
                "birdview": all_npc_bev,
                "state":    all_npc_state.astype(np.float32),
            }
            npc_input = {
                k: torch.as_tensor(v).to(self.device) for k, v in npc_input.items()
            }

            actions, values, log_prob, mu, sigma, features, cnn_feature = (
                self._policy.forward(npc_input, deterministic=True, clip_action=True)
            )
            actions = np.asarray(actions, dtype=np.float32)

            i = 0
            for npc in self.npc_vehicle_agents:
                if not npc.get("alive", True):
                    continue

                veh   = npc["veh"]
                acc   = actions[i][0]
                steer = actions[i][1]
                if acc >= 0.0:
                    throttle, brake = acc, 0.0
                else:
                    throttle, brake = 0.0, np.abs(acc)

                throttle = np.clip(throttle, 0, 1)
                brake    = np.clip(brake,    0, 1)
                steer    = np.clip(steer,   -1, 1)

                veh.update(steer, throttle, brake, dt)
                i += 1

        return collision_detected

    # ==================================================================
    # BEV 서피스 업데이트
    # ==================================================================
    def _update_bev_surfs(self, ref_vehicle=None):
        is_ego_mode = (ref_vehicle is None)
        if is_ego_mode:
            ref_vehicle = self.ego_vehicle

        ref_path, ref_path_idx = self._resolve_ref_path(ref_vehicle)

        ego_pixel_x, ego_pixel_y = world_to_pixel(
            ref_vehicle.x, ref_vehicle.y,
            self.OFFSET_X, self.OFFSET_Y, self.PPM,
        )
        ego_yaw_rad = ref_vehicle.yaw

        collision_ego_local_mask = make_ego_local_mask(
            canvas_h=self.OUT_SIZE, canvas_w=self.OUT_SIZE,
            ppm=self.PPM, yaw_rad=ego_yaw_rad,
            move_px=self.move_px, scale=1.15,
        )


        if ref_vehicle.layer > 0:
            das_local_bev = make_local_bev_ego_aligned(
                self.global_high_das_mask,
                center_px=(ego_pixel_x, ego_pixel_y),
                ego_yaw_rad=ego_yaw_rad,
                out_size=self.OUT_SIZE, out_mpp=1.0 / self.PPM,
                pixels_per_meter=self.PPM, fill=0, move_px=self.move_px,
            )
            lane_local_bev = make_local_bev_ego_aligned(
                self.global_high_lane_mask,
                center_px=(ego_pixel_x, ego_pixel_y),
                ego_yaw_rad=ego_yaw_rad,
                out_size=self.OUT_SIZE, out_mpp=1.0 / self.PPM,
                pixels_per_meter=self.PPM, fill=0, move_px=self.move_px,
            )
        else:
            das_local_bev = make_local_bev_ego_aligned(
                self.global_das_mask,
                center_px=(ego_pixel_x, ego_pixel_y),
                ego_yaw_rad=ego_yaw_rad,
                out_size=self.OUT_SIZE, out_mpp=1.0 / self.PPM,
                pixels_per_meter=self.PPM, fill=0, move_px=self.move_px,
            )
            lane_local_bev = make_local_bev_ego_aligned(
                self.global_lane_mask,
                center_px=(ego_pixel_x, ego_pixel_y),
                ego_yaw_rad=ego_yaw_rad,
                out_size=self.OUT_SIZE, out_mpp=1.0 / self.PPM,
                pixels_per_meter=self.PPM, fill=0, move_px=self.move_px,
            )

        # ── 경로 레이어 ──────────────────────────────────────────────
        path_local_bev = self._render_path_local_bev(
            ref_path, ref_path_idx,
            ego_pixel_x, ego_pixel_y, ego_yaw_rad,
        )

        # ── 히스토리 BEV ─────────────────────────────────────────────

        vehicle_bev_history, veh_collided = self.dynamic_map_manager.get_vehicle_bev_history(
            ego_pixel_x, ego_pixel_y, ref_vehicle.z, ego_yaw_rad, ref_vehicle.id,
            self.OUT_SIZE, self.PPM, self.move_px,
            ego_layer=ref_vehicle.layer,
            height_threshold=2.0,
            include_opposite_layer=False,
            return_collision=True,
        )
        ped_bev_history = self.dynamic_map_manager.get_ped_bev_history(
            ego_pixel_x, ego_pixel_y,
            ref_vehicle.z,
            ego_yaw_rad,
            self.OUT_SIZE, self.PPM, self.move_px,
        )
        # ── 충돌 체크 ────────────────────────────────────────────────

        ped_collided = check_ego_collision(
            collision_ego_local_mask, ped_bev_history[0], overlap_px_th=1,
        )
        is_collision = False
        collision_target = 'none'
        if veh_collided:
            is_collision     = True
            collision_target = "veh"

        if ped_collided:
            is_collision     = True
            collision_target = "ped"

        is_out_of_road, das_out_ratio = check_ego_out_of_road(
            collision_ego_local_mask, das_local_bev, out_of_road_ratio_th=0.80,
        )
        self.collision_info = {
            "is_collision":     is_collision,
            "collision_target": collision_target,
            "is_out_of_road":  is_out_of_road,
            "das_out_ratio":   das_out_ratio,
        }

        # ── 신호등 히스토리 BEV ──────────────────────────────────────
        tl_bev_history = self.get_tl_history_bev(
            ego_pixel_x, ego_pixel_y, ego_yaw_rad,
        )

        # ── 최종 마스크 스택 (C, H, W) ───────────────────────────────
        input_masks = np.stack(
            (
                das_local_bev, path_local_bev, lane_local_bev,
                *vehicle_bev_history,
                *ped_bev_history,
                *tl_bev_history,
            ),
            axis=0,
        )
        self.bev_surfs["input"] = input_masks

        if is_ego_mode:
            self.input_vis_rgb = make_input_vis_rgb(
                das_local_bev, path_local_bev, lane_local_bev,
                vehicle_bev_history, ped_bev_history, tl_bev_history,
            )
            return is_collision, is_out_of_road
        else:
            return is_collision, is_out_of_road, input_masks

    # ==================================================================
    # 동적 맵 갱신
    # ==================================================================
    def step_update_dynamic_maps(self) -> None:
        alive_npcs = [
            n for n in self.npc_vehicle_agents
            if n.get("alive", True)
        ]

        self.all_vehicle_agents = [{"veh": self.ego_vehicle, "alive": True}]
        self.all_vehicle_agents += alive_npcs

        self.dynamic_map_manager.push(
            vehicle_agents=self.all_vehicle_agents,
            ped_agents=self.ped_agents,
            offset_x=self.OFFSET_X,
            offset_y=self.OFFSET_Y,
            ppm=self.PPM,
            veh_len_m=VEHICLE_LENGTH_M,
            veh_wid_m=VEHICLE_WIDTH_M,
        )

    # ==================================================================
    # 신호등
    # ==================================================================
    def _build_tl_patch_cache(self, thickness=6):
        self._tl_thickness = int(thickness)
        cache = {}

        H, W = self.current_tl_mask.shape[:2]

        for tl_id, info in self.tl_stoplines.items():
            left = info["left"]
            right = info["right"]

            x1 = int(round((left.x  - self.OFFSET_X) * self.PPM))
            y1 = int(round((left.y  - self.OFFSET_Y) * self.PPM))
            x2 = int(round((right.x - self.OFFSET_X) * self.PPM))
            y2 = int(round((right.y - self.OFFSET_Y) * self.PPM))

            # 화면 밖 완전 제외
            if (x1 < 0 and x2 < 0) or (x1 >= W and x2 >= W) or (y1 < 0 and y2 < 0) or (y1 >= H and y2 >= H):
                continue

            cache[int(tl_id)] = {
                "pt1": (x1, y1),
                "pt2": (x2, y2),
            }

        return cache
    # ★★★ 핵심 수정: 단채널 마스크에 TL_VALUE_MAP 스칼라 값 사용 ★★★

    def _render_full_snapshot_into_mask(self, snapshot, out_mask):
        """전체 신호등 상태를 마스크에 한번에 그린다."""
        out_mask[...] = 0
        for tl_id, patch in self.tl_patch_cache.items():
            state = snapshot.get(tl_id, "R")
            value = int(TL_VALUE_MAP.get(state, np.uint8(255)))
            cv2.line(
                out_mask,
                patch["pt1"], patch["pt2"],
                value,                          # ★ 스칼라 (80/170/255)
                self._tl_thickness,
                lineType=cv2.LINE_AA,
            )

    def _update_changed_tl_mask(self, prev_snapshot, new_snapshot):
        """변경된 신호등만 마스크에서 갱신한다."""
        for tl_id, patch in self.tl_patch_cache.items():
            prev_state = prev_snapshot.get(tl_id, "R")
            new_state  = new_snapshot.get(tl_id, "R")
            if prev_state == new_state:
                continue

            # 이전 선 지우기 (0으로 덮기)
            cv2.line(
                self.current_tl_mask,
                patch["pt1"], patch["pt2"],
                0,                              # ★ 스칼라 0
                self._tl_thickness,
                lineType=cv2.LINE_AA,
            )
            # 새 상태 값으로 그리기
            new_value = int(TL_VALUE_MAP.get(new_state, np.uint8(255)))
            cv2.line(
                self.current_tl_mask,
                patch["pt1"], patch["pt2"],
                new_value,                      # ★ 스칼라 (80/170/255)
                self._tl_thickness,
                lineType=cv2.LINE_AA,
            )

    def step_traffic_lights(self, dt):
        prev_snapshot = self.current_tl_snapshot
        new_snapshot  = self.tl_manager.step(dt)
        self._update_changed_tl_mask(prev_snapshot, new_snapshot)
        self.current_tl_snapshot = new_snapshot
        self.tl_mask_history.append(self.current_tl_mask.copy())

    def _get_hist_mask(self, hist_idx):
        if len(self.tl_mask_history) == 0:
            return None
        idx = len(self.tl_mask_history) + hist_idx if hist_idx < 0 else hist_idx
        idx = max(0, min(idx, len(self.tl_mask_history) - 1))
        return self.tl_mask_history[idx]

    def get_tl_history_bev(self, cx, cy, ref_yaw):
        tl_bev_list = []
        for hist_idx in (-1, -5, -10, -15):
            global_tl_mask = self._get_hist_mask(hist_idx)
            if global_tl_mask is None:
                tl_bev_list.append(
                    np.zeros((self.OUT_SIZE, self.OUT_SIZE), dtype=np.uint8)
                )
                continue
            traffic_stop = make_local_bev_ego_aligned(
                global_tl_mask, (cx, cy), ref_yaw,
                out_size=self.OUT_SIZE, out_mpp=1.0 / self.PPM,
                pixels_per_meter=self.PPM, fill=0, move_px=self.move_px,
            )
            tl_bev_list.append(traffic_stop)
        return tl_bev_list

    # ==================================================================
    # 추적 오차
    # ==================================================================
    def _compute_tracking_errors(self, closest_idx: int):
        i0 = max(0, closest_idx - 1)
        i1 = min(len(self.ego_path) - 1, closest_idx + 1)

        p0 = np.array(self.ego_path[i0][:2], dtype=np.float32)
        p1 = np.array(self.ego_path[i1][:2], dtype=np.float32)
        p  = np.array([self.ego_vehicle.x, self.ego_vehicle.y], dtype=np.float32)

        v    = p1 - p0
        vv   = float(np.dot(v, v)) + 1e-6
        t    = np.clip(float(np.dot(p - p0, v) / vv), 0.0, 1.0)
        proj = p0 + t * v

        cte_mag    = float(np.linalg.norm(p - proj))
        cross_z    = float(v[0] * (p[1] - proj[1]) - v[1] * (p[0] - proj[0]))
        signed_cte = cte_mag if cross_z >= 0.0 else -cte_mag

        path_yaw      = to_rad_if_needed(self.ego_path[closest_idx][3])
        heading_error = normalize_angle_rad(path_yaw - self.ego_vehicle.yaw)
        return signed_cte, heading_error, path_yaw

    # ==================================================================
    # Info
    # ==================================================================
    def _get_info(self, success, collision, terminated, truncated):
        return {
            "town":       self.town,
            "step":       self.step_count,
            "path_idx":   int(self.path_idx),
            "road":       self.ego_path[self.path_idx][9].name,
            "success":    bool(success),
            "collision":  bool(collision),
            "speed":      float(self.ego_vehicle.v),
            "target_v":   float(self._last_target_v),
            "cte":        float(self._last_cte),
            "terminated": bool(terminated),
            "truncated":  bool(truncated),
        }

    # ==================================================================
    # 렌더링 (zoom / pan 지원)
    # ==================================================================
    def render(self):
        if not self.vis or self.render_mode != "human":
            return
        self._init_renderer_if_needed()

        self._handle_viewer_input()

        SCREEN_RES = self.SCREEN_RES
        PANEL_W    = self.PANEL_W
        vw         = self._viewer

        ego_px, ego_py = world_to_pixel(
            self.ego_vehicle.x, self.ego_vehicle.y,
            self.OFFSET_X, self.OFFSET_Y, self.PPM,
        )

        if vw.follow_ego:
            vw.center_x = float(ego_px)
            vw.center_y = float(ego_py)

        view_w_px = SCREEN_RES / vw.zoom
        view_h_px = SCREEN_RES / vw.zoom

        cam_x = vw.center_x - view_w_px / 2
        cam_y = vw.center_y - view_h_px / 2

        cam_x = max(0, min(cam_x, self.MAP_W - view_w_px))
        cam_y = max(0, min(cam_y, self.MAP_H - view_h_px))

        cam_x_int = int(cam_x)
        cam_y_int = int(cam_y)
        view_w_int = int(min(view_w_px, self.MAP_W - cam_x_int))
        view_h_int = int(min(view_h_px, self.MAP_H - cam_y_int))

        if view_w_int < 1 or view_h_int < 1:
            return

        camera_rect = pygame.Rect(cam_x_int, cam_y_int, view_w_int, view_h_int)

        sub_map    = self._full_map_surface.subsurface(camera_rect)
        scaled_map = pygame.transform.smoothscale(sub_map, (SCREEN_RES, SCREEN_RES))
        self._screen.blit(scaled_map, (0, 0))

        # ── 신호등 오버레이 ──────────────────────────────────────────
        tl_region = self.current_tl_mask[cam_y_int:cam_y_int + view_h_int,
                                         cam_x_int:cam_x_int + view_w_int]
        if np.any(tl_region > 0):
            tl_rgba = np.zeros((view_h_int, view_w_int, 4), dtype=np.uint8)
            # ★ TL_VALUE_MAP 스칼라 값과 매칭
            g_mask = tl_region == 80    # TL_VALUE_MAP["G"]
            tl_rgba[g_mask] = (0, 200, 0, 200)
            y_mask = tl_region == 170   # TL_VALUE_MAP["Y"]
            tl_rgba[y_mask] = (255, 200, 0, 200)
            r_mask = tl_region == 255   # TL_VALUE_MAP["R"]
            tl_rgba[r_mask] = (255, 0, 0, 200)

            tl_surf = pygame.image.frombuffer(
                tl_rgba.tobytes(), (view_w_int, view_h_int), "RGBA"
            )
            tl_scaled = pygame.transform.smoothscale(tl_surf, (SCREEN_RES, SCREEN_RES))
            self._screen.blit(tl_scaled, (0, 0))

        # ── 경로 타겟 포인트 ─────────────────────────────────────────
        tpx = self.ego_path_px[self.path_idx][0]
        tpy = self.ego_path_px[self.path_idx][1]
        scale_x = SCREEN_RES / view_w_int
        scale_y = SCREEN_RES / view_h_int
        pygame.draw.circle(
            self._screen, (255, 0, 0),
            (int((tpx - cam_x_int) * scale_x),
             int((tpy - cam_y_int) * scale_y)),
            8,
        )

        def to_screen(px, py):
            return (int((px - cam_x_int) * scale_x),
                    int((py - cam_y_int) * scale_y))

        # ── NPC 렌더링 ───────────────────────────────────────────────
        for npc in self.npc_vehicle_agents:
            if not npc.get("alive", True):
                continue
            veh       = npc["veh"]
            npx, npy  = world_to_pixel(
                veh.x, veh.y, self.OFFSET_X, self.OFFSET_Y, self.PPM,
            )
            margin = 50
            if (cam_x_int - margin <= npx <= cam_x_int + view_w_int + margin and
                    cam_y_int - margin <= npy <= cam_y_int + view_h_int + margin):
                nsx, nsy    = to_screen(npx, npy)
                angle_deg   = -math.degrees(veh.yaw)
                surf        = npc.get("surf", self.npc_base)
                rotated_npc = pygame.transform.rotozoom(surf, angle_deg, vw.zoom)
                npc_rect    = rotated_npc.get_rect(center=(nsx, nsy))
                self._screen.blit(rotated_npc, npc_rect.topleft)

        # ── ego 렌더링 ───────────────────────────────────────────────
        esx, esy      = to_screen(ego_px, ego_py)
        ego_angle_deg = -math.degrees(self.ego_vehicle.yaw)
        rotated_ego   = pygame.transform.rotozoom(self.car_base, ego_angle_deg, vw.zoom)
        ego_rect      = rotated_ego.get_rect(center=(esx, esy))
        self._screen.blit(rotated_ego, ego_rect.topleft)

        # ── 사이드 패널 ──────────────────────────────────────────────
        panel_x = SCREEN_RES
        self._screen.fill(
            (30, 30, 30),
            rect=pygame.Rect(panel_x, 0, PANEL_W, SCREEN_RES),
        )

        if hasattr(self, "input_vis_rgb"):
            blit_rgb_to_surface(self._bev_vis_surfs["input"], self.input_vis_rgb)

        TILE         = self.OUT_SIZE
        panel_blit_x = panel_x + (PANEL_W - TILE) // 2
        panel_blit_y = max(0, (SCREEN_RES - 5 * TILE) // 2)
        self._screen.blit(self._bev_vis_surfs["input"], (panel_blit_x, panel_blit_y))

        active_tl_id = self.ego_path[self.path_idx][8]
        st           = self.tl_manager.state(active_tl_id)
        txet         = [st if st in ("Y", "R") else "G"]
        txet2        = ["R"] if self.planner.traffic else ["G"]

        hud_lines = [
            f"Step: {self.step_count}",
            f"Speed: {float(self.ego_vehicle.v):.2f} m/s",
            f"steer: {float(self.steer):.2f}",
            f"throttle: {float(self.throttle):.2f}",
            f"brake: {float(self.brake):.2f}",
            f"traffic L: {txet}",
            f"traffic in: {txet2}",
            f"total_reward: {float(self.total_reward):.2f}",
            f"Path Idx: {self.path_idx / len(self.ego_path) * 100:.2f}%",
            f"road option: {self.ego_path[self.path_idx][9].name}",
            f"zoom: {vw.zoom:.2f}x",
            f"follow: {'ON' if vw.follow_ego else 'OFF'}",
        ]
        y_offset = 15
        for line in hud_lines:
            txt_surf = self._hud_font.render(line, True, (255, 255, 0))
            self._screen.blit(txt_surf, (15, y_offset))
            y_offset += 25

        pygame.display.flip()
        self._clock.tick(self.RENDER_FPS)

    # ------------------------------------------------------------------
    # 뷰어 입력 처리 (zoom / pan)
    # ------------------------------------------------------------------
    def _handle_viewer_input(self) -> None:
        vw = self._viewer

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

            if event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    vw.zoom = min(vw.zoom * vw.zoom_step, vw.zoom_max)
                elif event.y < 0:
                    vw.zoom = max(vw.zoom / vw.zoom_step, vw.zoom_min)

        keys = pygame.key.get_pressed()

        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            vw.zoom = min(vw.zoom * vw.zoom_step, vw.zoom_max)
        if keys[pygame.K_MINUS]:
            vw.zoom = max(vw.zoom / vw.zoom_step, vw.zoom_min)

        pan = vw.pan_speed / vw.zoom
        if keys[pygame.K_LEFT]:
            vw.center_x -= pan
            vw.follow_ego = False
        if keys[pygame.K_RIGHT]:
            vw.center_x += pan
            vw.follow_ego = False
        if keys[pygame.K_UP]:
            vw.center_y -= pan
            vw.follow_ego = False
        if keys[pygame.K_DOWN]:
            vw.center_y += pan
            vw.follow_ego = False

        if keys[pygame.K_f]:
            vw.follow_ego = True

    # ==================================================================
    # 종료
    # ==================================================================
    def close(self):
        if self._pygame_inited:
            pygame.quit()
            self._pygame_inited = False

    # ==================================================================
    # 유틸리티
    # ==================================================================
    def _resolve_ref_path(self, ref_vehicle) -> Tuple[list, int]:
        if ref_vehicle is self.ego_vehicle:
            return self.ego_path, self.path_idx

        ref_npc_agent = next(
            (n for n in self.npc_vehicle_agents if n["veh"] is ref_vehicle),
            None,
        )
        if ref_npc_agent:
            return ref_npc_agent["path"], ref_npc_agent["idx"]
        return self.ego_path, self.path_idx

    def _get_other_vehicle_agents(self, ref_vehicle) -> list:
        others = []
        if ref_vehicle is not self.ego_vehicle:
            others.append({"id": 0, "veh": self.ego_vehicle, "alive": True})
        others += [
            n for n in self.npc_vehicle_agents
            if n.get("alive", True) and n["veh"] is not ref_vehicle
        ]
        return others

    def _render_path_local_bev(
        self,
        ref_path:     list,
        ref_path_idx: int,
        ego_pixel_x:  float,
        ego_pixel_y:  float,
        ego_yaw_rad:  float,
    ) -> np.ndarray:
        path_local_bev = np.zeros((self.OUT_SIZE, self.OUT_SIZE), dtype=np.uint8)
        start_idx = max(0, ref_path_idx - 8)
        end_idx   = min(len(ref_path), ref_path_idx + 50)
        local_pts = []

        for pt in ref_path[start_idx:end_idx]:
            global_px, global_py = world_to_pixel(
                pt[0], pt[1],
                self.OFFSET_X, self.OFFSET_Y, self.PPM,
            )
            local_u, local_v = global_pixel_to_local_bev(
                global_px, global_py,
                ego_pixel_x, ego_pixel_y, ego_yaw_rad,
                out_size=self.OUT_SIZE, out_mpp=1.0 / self.PPM,
                pixels_per_meter=self.PPM, move_px=self.move_px,
            )
            if 0 <= local_u < self.OUT_SIZE and 0 <= local_v < self.OUT_SIZE:
                local_pts.append([local_u, local_v])

        if len(local_pts) >= 2:
            cv2.polylines(
                path_local_bev,
                [np.array(local_pts, dtype=np.int32).reshape((-1, 1, 2))],
                isClosed=False, color=255, thickness=12,
            )
        elif len(local_pts) == 1:
            cv2.circle(path_local_bev, tuple(local_pts[0]), 6, 255, -1)

        return path_local_bev
