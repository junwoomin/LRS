import os
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2

from sim.traffic import *
from sim.utility import *
from sim.routeplanner import *
from sim.driving import *

import gymnasium as gym
from gymnasium import spaces




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


class BEVPathFollowEnv(gym.Env):

    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        town               = "Town01",
        vis                = False,
        render_mode        = "none",
        out_size           = 192,
        panel_update_every = 3,
        PPM                = 5.0,
        render_fps         = 60,
        sim_steps_per_frame = 3,
        num_routes         = 20,
        waypoint_sample_dist = 1000.0,
        cruise_speed       = 20.0,
        loop_route         = False,
        end_on_route_end   = True,
        max_steps          = 2000,
        reward_cfg         = None,
        seed               = None,
        logger             = None,
        FIXED_DT           = 0.01,
    ):
        super().__init__()
        self.logger = logger

        self.town        = town
        self.vis         = bool(vis)
        self.render_mode = render_mode if render_mode in ["human", "none"] else "none"

        self.RENDER_FPS          = int(render_fps)
        self.SIM_STEPS_PER_FRAME = int(sim_steps_per_frame)
        self.FIXED_DT            = FIXED_DT
        self.move_px             = 40
        self.PPM                 = float(PPM)

        self.OUT_SIZE           = int(out_size)
        self.PANEL_UPDATE_EVERY = int(panel_update_every)

        self.NUM_ROUTES            = int(num_routes)
        self.WAYPOINT_SAMPLE_DIST  = float(waypoint_sample_dist)
        self.CRUISE_SPEED          = float(cruise_speed)
        self.LOOP_ROUTE            = bool(loop_route)
        self.END_ON_ROUTE_END      = bool(end_on_route_end)

        self.max_steps  = int(max_steps)
        self.reward_cfg = reward_cfg if reward_cfg is not None else RewardConfig()

        self._np_random = np.random.RandomState(seed)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.curr_target_speed_kmh   = float(self.reward_cfg.target_speed_kmh)
        self.curr_min_speed_kmh      = float(self.reward_cfg.min_speed_kmh)
        self.curr_max_distance       = float(self.reward_cfg.max_distance)
        self.curr_max_angle_center_lane = float(self.reward_cfg.max_angle_center_lane)

        self.roadoption_to_idx = {-1: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}

        self.observation_space = spaces.Dict({
            "bev":   spaces.Box(low=0.0, high=1.0, shape=(15, self.OUT_SIZE, self.OUT_SIZE), dtype=np.float32),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
        })

        if not hasattr(self, 'car_base'):
            self.car_base = pygame.Surface((4.69 * 20, 2.0 * 20), pygame.SRCALPHA)
            self.car_base.fill((255, 50, 50))
            self.npc_base = pygame.Surface((4.69 * 20, 2.0 * 20), pygame.SRCALPHA)
            self.npc_base.fill((50, 150, 255))

        self._load_static_assets()

        self._pygame_inited   = False
        self._screen          = None
        self._clock           = None
        self._hud_font        = None
        self._full_map_surface = None

        self._reset_episode_state()

    # ------------------------------------------------------------------
    # 초기화 / 로딩
    # ------------------------------------------------------------------
    def _load_static_assets(self) -> None:
        Town = self.town

        self.DATA20_DIR = f"sim_using_data/data20/{Town}"
        self.DATA5_DIR  = f"sim_using_data/data5/{Town}"

        self.PATHS20_PATH    = f"{self.DATA20_DIR}/{Town}.png"
        self.DAS20_PATH      = f"{self.DATA20_DIR}/das_full.png"
        self.LANE20_PATH     = f"{self.DATA20_DIR}/lane_full.png"
        self.STOP_MASK_PATH  = f"{self.DATA5_DIR}/stoplines.png"
        self.DAS5_MASK_PATH  = f"{self.DATA5_DIR}/das_full.png"
        self.LANE5_MASK_PATH = f"{self.DATA5_DIR}/lane_full.png"
        self.PATH5_MASK_PATH = f"{self.DATA5_DIR}/{Town}.png"
        self.OFFSET_PATH     = f"{self.DATA20_DIR}/world_offset.npy"
        self.XODR_PATH       = f"sim_using_data/Town/{Town}.xodr"

        if not os.path.exists(self.OFFSET_PATH):
            raise FileNotFoundError(f"Offset file not found: {self.OFFSET_PATH}")
        world_offset     = np.load(self.OFFSET_PATH)
        self.OFFSET_X, self.OFFSET_Y = float(world_offset[0]), float(world_offset[1])

        ped_json      = f"sim_using_data/Town/{Town}_ped_graph.json"
        self.ped_graph = PedGraph(ped_json, grid=2.0)

        paths_img = cv2.imread(self.PATHS20_PATH)
        das_img   = cv2.imread(self.DAS20_PATH)
        lane_img  = cv2.imread(self.LANE20_PATH)

        if paths_img is None: raise FileNotFoundError(f"Missing image: {self.PATHS20_PATH}")
        if das_img   is None: raise FileNotFoundError(f"Missing image: {self.DAS20_PATH}")
        if lane_img  is None: raise FileNotFoundError(f"Missing image: {self.LANE20_PATH}")

        self.MAP_H, self.MAP_W, _ = paths_img.shape

        self.das_mask20  = to_mask(das_img,   thr=10)
        self.lane_mask20 = to_mask(lane_img,  thr=10)
        self.path_mask20 = to_mask(paths_img, thr=10)

        canvas20 = np.zeros((self.MAP_H, self.MAP_W, 3), dtype=np.uint8)
        canvas20 = overlay_solid(canvas20, self.das_mask20,  (90,  90,  90),  alpha=1.0)
        canvas20 = overlay_solid(canvas20, self.lane_mask20, (255, 255, 255), alpha=1.0)
        canvas20 = overlay_solid(canvas20, self.path_mask20, (0,   255, 255), alpha=1.0)
        self.canvas20 = canvas20

        self.das_mask5  = cv2.imread(self.DAS5_MASK_PATH,  cv2.IMREAD_GRAYSCALE)
        self.lane_mask5 = cv2.imread(self.LANE5_MASK_PATH, cv2.IMREAD_GRAYSCALE)
        self.stop_mask5 = cv2.imread(self.STOP_MASK_PATH,  cv2.IMREAD_GRAYSCALE)

        if self.das_mask5  is None: raise FileNotFoundError(f"Missing mask: {self.DAS5_MASK_PATH}")
        if self.lane_mask5 is None: raise FileNotFoundError(f"Missing mask: {self.LANE5_MASK_PATH}")

        h, w = self.das_mask5.shape[:2]
        self.base_img = np.zeros((h, w, 3), dtype=np.uint8)

        with open(self.XODR_PATH, "r", encoding="utf-8") as f:
            xodr_content = f.read()
        self.world_map = carla.Map(f"{Town}", xodr_content)

        landmarks        = self.world_map.get_all_landmarks()
        self.tl_landmarks = [lm for lm in landmarks if is_traffic_light_landmark(lm)]

        self.tl_stoplines = {}
        for lm in self.tl_landmarks:
            stop_wp, loc_left, loc_right = get_stopline_vertices_from_landmark(self.world_map, lm)
            if stop_wp is not None:
                self.tl_stoplines[int(lm.id)] = {
                    "stop_wp": stop_wp,
                    "left":    loc_left,
                    "right":   loc_right,
                }

    def _init_renderer_if_needed(self) -> None:
        if not self.vis or self._pygame_inited:
            return

        os.environ.setdefault("SDL_VIDEODRIVER", "x11")
        pygame.init()
        self._pygame_inited = True

        self.SCREEN_RES = 1000
        self.PANEL_W    = 200

        self._hud_font  = pygame.font.SysFont("consolas", 18)
        self._screen    = pygame.display.set_mode((self.SCREEN_RES + self.PANEL_W, self.SCREEN_RES))
        pygame.display.set_caption(f"BEVPathFollowEnv - {self.town}")
        self._clock     = pygame.time.Clock()

        self._full_map_surface = pygame.surfarray.make_surface(self.canvas20.swapaxes(0, 1))
        self._bev_vis_surfs    = {"input": pygame.Surface((self.OUT_SIZE, self.OUT_SIZE))}

    # ------------------------------------------------------------------
    # 에피소드 상태
    # ------------------------------------------------------------------
    def _reset_episode_state(self) -> None:
        self.ego_path      = []
        self.ego_path_px20 = None
        self.path_idx      = 0
        self.current_lookahead = 7.0

        self.ego_vehicle = None
        self.npc_agents  = []
        self.ped_agents  = []
        self.cte_history = []

        self.planner = LongitudinalPlanner()

        self.tl_manager = TrafficLight_Manager(
            self.world_map, self.tl_landmarks,
            t_green=10.0, t_yellow=3.0, t_all_red=2.0
        )

        self.throttle = 0.0
        self.brake    = 0.0
        self.steer    = 0.0

        self.step_count         = 0
        self.prev_closest_idx   = 0
        self.prev_path_idx      = 0
        self.last_steer         = 0.0
        self.steer_cmd          = 0
        self.throttle_cmd       = 0
        self.brake_cmd          = 0

        self._last_target_v     = 0.0
        self._last_cte          = 0.0

        self.bev_surfs = {
            "input": np.zeros((self.OUT_SIZE, self.OUT_SIZE, 15), dtype=np.uint8),
        }

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
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

        # ego 경로 생성
        self.ego_path = []
        for _ in range(20):
            ego_results = generate_random_routes(
                self.world_map, self.das_mask5, self.OFFSET_X, self.OFFSET_Y,
                num_routes=1, sample_dist=4000.0, max_len=12000, ppm=5.0, invert_y=False
            )
            candidate = ego_results[0] if ego_results else []
            if len(candidate) >= 80:
                self.ego_path = candidate
                break

        if len(self.ego_path) < 80:
            raise RuntimeError("Valid ego route not found")

        # NPC 경로
        NUM_NPC   = 30
        npc_routes = generate_random_routes(
            self.world_map, self.das_mask5, self.OFFSET_X, self.OFFSET_Y,
            num_routes=NUM_NPC, sample_dist=4000.0, max_len=8000, ppm=5.0, invert_y=False
        )

        # 보행자 에이전트
        self.ped_agents = []
        for _ in range(50):
            p = PedAgent(
                self.ped_graph,
                speed=np.random.uniform(1.3, 2.4),
                reach=0.7,
                max_yaw_rate=8.0,
                jaywalk_cost=0.7,
            )
            if p.alive:
                self.ped_agents.append(p)

        self.ego_path_px20 = np.array(
            [world_to_pixel(x, y, self.OFFSET_X, self.OFFSET_Y, 20)
             for (x, y, _, _, _, _, _, _, _, _) in self.ego_path],
            dtype=np.int32
        )

        # 신호등 초기 이미지
        self.img_traffic = np.zeros_like(self.base_img, dtype=np.uint8)
        for lm in self.tl_landmarks:
            tl_id = int(lm.id)
            if tl_id not in self.tl_stoplines:
                continue
            state = self.tl_manager.state(tl_id)
            info  = self.tl_stoplines[tl_id]
            draw_stopline_from_vertices(
                self.img_traffic, info["left"], info["right"], state,
                self.OFFSET_X, self.OFFSET_Y, ppm=5.0, thickness=6
            )

        # ego 차량 생성
        self.ego_vehicle = BicycleModel(
            self.ego_path[0][0], self.ego_path[0][1],
            self.ego_path[0][2], self.ego_path[0][3]
        )
        self.ego_vehicle.v = float(INIT_SPEED) if "INIT_SPEED" in globals() else 0.0

        # NPC 차량 생성
        self.npc_agents = []
        ego_x, ego_y   = self.ego_vehicle.x, self.ego_vehicle.y
        MIN_SPAWN_DIST  = 6.0

        for k, route in enumerate(npc_routes, start=1):
            if len(route) < 2:
                continue

            npc_start_x, npc_start_y = route[0][0], route[0][1]
            if math.hypot(npc_start_x - ego_x, npc_start_y - ego_y) < MIN_SPAWN_DIST:
                continue

            too_close = any(
                math.hypot(npc_start_x - other["veh"].x, npc_start_y - other["veh"].y) < MIN_SPAWN_DIST
                for other in self.npc_agents
            )
            if too_close:
                continue

            npc_vehicle       = BicycleModel(npc_start_x, npc_start_y, route[0][2], route[0][3])
            npc_vehicle.v     = float(INIT_SPEED) if "INIT_SPEED" in globals() else 0.0
            npc_pid           = PID(0.8, 0.2, 0.05, out_min=-1.0, out_max=1.0)
            cruise_k          = self.CRUISE_SPEED * float(self._np_random.uniform(0.7, 1.1))

            self.npc_agents.append({
                "id":         k,
                "path":       route,
                "lookahead":  10.0,
                "veh":        npc_vehicle,
                "idx":        0,
                "pid":        npc_pid,
                "planner":    NPCLongitudinalPlanner(),
                "cruise":     cruise_k,
                "alive":      True,
                "last_steer": 0.0,
            })

        closest_idx           = find_closest_index(self.ego_path, self.ego_vehicle.x, self.ego_vehicle.y)
        self.prev_closest_idx = closest_idx
        self.prev_path_idx    = self.path_idx

        self._update_bev_surfs()

        if self.vis and self.render_mode == "human":
            self._init_renderer_if_needed()

        val = self.ego_path[0][9].value
        idx = self.roadoption_to_idx[val]
        self.road_op       = np.zeros(7, dtype=np.float32)
        self.road_op[idx]  = 1.0

        self.current_all_vehicles = [{"id": 0, "veh": self.ego_vehicle}] + [
            {"id": n["id"], "veh": n["veh"]} for n in self.npc_agents if n["alive"]
        ]

        obs  = self._get_obs()
        info = self._get_info(success=False, collision=False, terminated=False, truncated=False)
        self.stationary_steps = 0
        return obs, info

    def step(self, action):
        terminated = False
        truncated  = False

        acc   = action[0][0]
        steer = action[0][1]

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

        for _ in range(self.SIM_STEPS_PER_FRAME):
            dt = self.FIXED_DT
            self.tl_manager.step(dt)

            self.ego_vehicle.update(self.steer, self.throttle, self.brake, dt)

            self.path_idx = find_lookahead_index(
                self.ego_path, self.path_idx,
                self.ego_vehicle.x, self.ego_vehicle.y,
                self.current_lookahead,
            )

            self.collision_detected = self._update_npcs_and_check_collision(dt)
            if self.collision_detected:
                break
            if self._is_outside_das(self.ego_vehicle.x, self.ego_vehicle.y):
                self.DAS_collision_detected = True
                break

            closest_idx_now = find_closest_index(self.ego_path, self.ego_vehicle.x, self.ego_vehicle.y)
            if remaining_distance(self.ego_path, closest_idx_now) < 3.0:
                self.success = True
                break

        val = self.ego_path[self.path_idx][9].value
        idx = self.roadoption_to_idx[val]
        self.road_op      = np.zeros(7, dtype=np.float32)
        self.road_op[idx] = 1.0
        self.ego_vehicle.z = self.ego_path[self.path_idx][2]

        self.veh_collision, self.das_collision = self._update_bev_surfs()

        self.current_all_vehicles = [{"id": 0, "veh": self.ego_vehicle}] + [
            {"id": n["id"], "veh": n["veh"]} for n in self.npc_agents if n["alive"]
        ]

        reward, terminated, truncated = self.get_reward()

        self.step_count        += 1
        self.prev_closest_idx   = self.closest_idx
        self.total_reward      += float(reward)

        obs  = self._get_obs()
        info = self._get_info(
            success=self.success, collision=self.collision_detected,
            terminated=terminated, truncated=truncated
        )
        return obs, float(reward), (terminated or truncated), info

    def _compute_guidance_steer(self, closest_idx: int, signed_cte: float, heading_error: float) -> float:
        speed_ms  = max(0.1, float(self.ego_vehicle.v))
        speed_kmh = speed_ms * 3.6
        lookahead = float(np.clip(4.5 + 0.12 * speed_kmh, 4.5, 10.0))
        self.current_lookahead = lookahead

        target_idx = find_lookahead_index(
            self.ego_path, closest_idx,
            self.ego_vehicle.x, self.ego_vehicle.y, lookahead
        )
        tx, ty     = self.ego_path[target_idx][0], self.ego_path[target_idx][1]
        pure_delta = pure_pursuit_control(self.ego_vehicle, tx, ty, lookahead)

        stanley_gain  = 0.85
        stanley_delta = heading_error - math.atan2(stanley_gain * signed_cte, speed_ms + 1.0)

        blend = float(self.reward_cfg.guidance_blend)
        delta = blend * pure_delta + (1.0 - blend) * stanley_delta
        return float(np.clip(delta / max(1e-6, self.ego_vehicle.max_steer), -1.0, 1.0))

    def get_reward(self):
        rc        = self.reward_cfg
        reward    = 0.0
        terminated = False
        truncated  = False
        self.reason = None

        env_speed_ms   = float(self.ego_vehicle.v)
        self.closest_idx = find_closest_index(self.ego_path, self.ego_vehicle.x, self.ego_vehicle.y)
        signed_cte, heading_error, _ = self._compute_tracking_errors(self.closest_idx)
        cte_rate = signed_cte - float(self._last_cte)

        target_v_ms             = self._compute_target_speed_ms(self.closest_idx)
        self._last_target_v     = target_v_ms * 3.6
        self._last_guidance_steer = self._compute_guidance_steer(self.closest_idx, signed_cte, heading_error)

        progress_idx   = min(max(0, self.closest_idx - self.prev_closest_idx), 2)
        lane_gate      = max(0.0, 1.0 - abs(signed_cte) / max(1e-6, self.curr_max_distance))
        progress_reward = float(progress_idx) * lane_gate

        center_reward  = math.exp(-abs(signed_cte)    / 0.75)
        heading_reward = math.exp(-abs(heading_error) / np.deg2rad(10.0))
        speed_reward   = math.exp(-abs(env_speed_ms - target_v_ms) / 2.5)

        cte_growth_penalty = 0.0
        if abs(signed_cte) > abs(self._last_cte):
            cte_growth_penalty = rc.cte_growth_weight * min(1.0, abs(cte_rate) / 0.5)

        reward += rc.progress_weight * progress_reward
        reward += rc.center_weight   * center_reward
        reward += rc.heading_weight  * heading_reward
        reward += rc.speed_weight    * speed_reward
        reward -= cte_growth_penalty

        if abs(signed_cte) < rc.center_bonus_cte_m and \
           abs(heading_error) < np.deg2rad(rc.center_bonus_heading_deg):
            reward += rc.center_bonus

        self._last_cte            = float(signed_cte)
        self._last_heading_error  = float(heading_error)
        self._last_cte_rate       = float(cte_rate)
        self._last_progress_ratio = float(self.closest_idx / max(1, len(self.ego_path) - 1))

        if self.reason is not None and self.logger is not None:
            self.logger.info(f"[Step {self.step_count}] {self.reason}")

        return reward, terminated, truncated

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

    # ------------------------------------------------------------------
    # NPC 업데이트
    # ------------------------------------------------------------------
    def _update_npcs_and_check_collision(self, dt):
        collision_detected = False
        ego_pos            = (self.ego_vehicle.x, self.ego_vehicle.y)
        ego_yaw_rad        = self.ego_vehicle.yaw

        current_all_vehicles = [{"id": 0, "veh": self.ego_vehicle}] + [
            {"id": n["id"], "veh": n["veh"]} for n in self.npc_agents if n.get("alive", True)
        ]

        for ped_npc in self.ped_agents:
            ped_npc.step(dt)

        for npc in self.npc_agents:
            if not npc.get("alive", True):
                continue

            veh  = npc["veh"]
            path = npc["path"]
            idx  = int(npc["idx"])
            pid  = npc["pid"]
            planner_npc = npc["planner"]

            npc_pos = (veh.x, veh.y)
            dist    = math.hypot(ego_pos[0] - npc_pos[0], ego_pos[1] - npc_pos[1])
            if dist < 5.0:
                if is_colliding(
                    ego_pos, ego_yaw_rad,
                    npc_pos, veh.yaw,
                    ego_size=(4.69, 2.0), other_size=(4.69, 2.0),
                ):
                    collision_detected = True
                    break

            current_lookahead2 = float(npc.get("lookahead", 10.0))
            idx = find_lookahead_index(path, idx, veh.x, veh.y, current_lookahead2)

            if idx >= len(path) - int(current_lookahead2 + 2):
                if self.LOOP_ROUTE:
                    start_point = carla.Location(x=float(veh.x), y=float(veh.y), z=0.0)
                    new_routes  = generate_random_routes_roop(
                        self.world_map, start_point,
                        num_routes=1, sample_dist=self.WAYPOINT_SAMPLE_DIST, max_len=4000
                    )
                    npc["path"] = new_routes[0] if new_routes else path
                    path        = npc["path"]
                    idx         = 0
                    veh.x, veh.y, veh.yaw = path[0][0], path[0][1], path[0][3]
                    veh.v = float(INIT_SPEED) if "INIT_SPEED" in globals() else 0.0
                    pid.reset()
                else:
                    npc["alive"] = False
                    continue

            ntx, nty   = path[idx][0], path[idx][1]
            ndelta     = pure_pursuit_control(veh, ntx, nty, current_lookahead2)
            ndelta_clamped = clamp(ndelta, -veh.max_steer, veh.max_steer)

            n_closest   = find_closest_index(path, veh.x, veh.y)
            ntarget_v   = planner_npc.compute_target_speed(
                npc["path"], n_closest, veh,
                current_all_vehicles, self.ped_agents, npc["cruise"], self.tl_manager
            )

            if ntarget_v < 0.1 or getattr(planner_npc, "is_waiting", False):
                nthrottle, nbrake = 0.0, 1.0
                pid.reset()
            else:
                nerr_v = ntarget_v - veh.v
                nu     = pid.step(nerr_v, dt)
                if nu >= 0:
                    nthrottle = clamp(nu,  0.0, 1.0)
                    nbrake    = 0.0
                else:
                    nthrottle = 0.0
                    nbrake    = clamp(-nu, 0.0, 1.0)

            veh.update(ndelta_clamped, nthrottle, nbrake, dt)
            npc["idx"]        = idx
            npc["last_steer"] = ndelta_clamped

        return collision_detected

    def _is_outside_das(self, x: float, y: float) -> bool:
        px, py = world_to_pixel(x, y, self.OFFSET_X, self.OFFSET_Y, self.PPM)
        h, w   = self.das_mask5.shape[:2]
        if px < 0 or py < 0 or px >= w or py >= h:
            return True
        v      = self.das_mask5[int(py), int(px)]
        return not (v > 0)

    # ------------------------------------------------------------------
    # BEV 생성 (ref_vehicle 기반)
    # ------------------------------------------------------------------
    def _update_bev_surfs(self, ref_vehicle=None):
        if ref_vehicle is None:
            ref_vehicle = self.ego_vehicle

        ref_x   = ref_vehicle.x
        ref_y   = ref_vehicle.y
        ref_yaw = ref_vehicle.yaw

        # ref_vehicle의 경로 및 경로 인덱스 결정
        if ref_vehicle is self.ego_vehicle:
            ref_path      = self.ego_path
            ref_path_idx  = self.path_idx
        else:
            ref_npc      = next((n for n in self.npc_agents if n["veh"] is ref_vehicle), None)
            ref_path     = ref_npc["path"] if ref_npc else self.ego_path
            ref_path_idx = ref_npc["idx"]  if ref_npc else self.path_idx

        # BEV 중심 픽셀
        cx5, cy5 = world_to_pixel(ref_x, ref_y, self.OFFSET_X, self.OFFSET_Y, self.PPM)

        # ref를 제외한 다른 차량 목록
        other_agents = []
        if ref_vehicle is not self.ego_vehicle:
            other_agents.append({"id": 0, "veh": self.ego_vehicle, "alive": True})
        other_agents += [n for n in self.npc_agents if n.get("alive", True) and n["veh"] is not ref_vehicle]

        # --- 정적 레이어 ---
        das = make_local_bev_ego_aligned(
            self.das_mask5, (cx5, cy5), ref_yaw,
            out_size=self.OUT_SIZE, out_mpp=1 / self.PPM,
            pixels_per_meter=self.PPM, fill=0, move_px=self.move_px
        )
        lane = make_local_bev_ego_aligned(
            self.lane_mask5, (cx5, cy5), ref_yaw,
            out_size=self.OUT_SIZE, out_mpp=1 / self.PPM,
            pixels_per_meter=self.PPM, fill=0, move_px=self.move_px
        )

        # --- 경로 레이어 ---
        imgs  = np.zeros((self.OUT_SIZE, self.OUT_SIZE), dtype=np.uint8)
        start = max(0, ref_path_idx - 7)
        end   = min(len(ref_path), ref_path_idx + 50)
        poly_pts = []
        for pt in ref_path[start:end]:
            wx, wy  = pt[0], pt[1]
            px, py  = world_to_pixel(wx, wy, self.OFFSET_X, self.OFFSET_Y, self.PPM)
            u, v    = global_pixel_to_local_bev(
                px, py, cx5, cy5, ref_yaw,
                out_size=self.OUT_SIZE, out_mpp=1 / self.PPM,
                pixels_per_meter=self.PPM, move_px=self.move_px
            )
            if 0 <= u < self.OUT_SIZE and 0 <= v < self.OUT_SIZE:
                poly_pts.append([u, v])

        if len(poly_pts) >= 2:
            cv2.polylines(imgs, [np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))],
                          False, 255, thickness=12)
        elif len(poly_pts) == 1:
            cv2.circle(imgs, tuple(poly_pts[0]), 6, 255, -1)

        path_bev = imgs.copy()

        # --- 차량 현재 마스크 ---
        veh_merged, ego_mask_bin, npc_mask_bin = make_vehicle_bev_ego_aligned(
            ref_vehicle, other_agents,
            out_size=self.OUT_SIZE, out_mpp=1 / self.PPM,
            veh_len_m=VEHICLE_LENGTH_M, veh_wid_m=VEHICLE_WIDTH_M,
            move_px=self.move_px
        )

        # 충돌 체크
        veh_collision, das_collision, info = bev_collision_check(
            ego_mask_bin, npc_mask_bin, das,
            overlap_px_th=1, das_out_ratio_th=0.30
        )
        self.bev_collision_info = info


        all_other_for_hist = []
        if ref_vehicle is not self.ego_vehicle:
            all_other_for_hist.append({"id": 0, "veh": self.ego_vehicle})
        all_other_for_hist += [n for n in self.npc_agents if n["veh"] is not ref_vehicle]

        npc_by_hist = build_npc_agents_by_history(all_other_for_hist, [-1, -5, -10, -15])
        ped_by_hist = build_ped_agents_by_history(self.ped_agents,     [-1, -5, -10, -15])

        c_vehicle_history = []
        c_walker_history  = []

        for hist_idx in [-1, -5, -10, -15]:
            mask_npc, _, _ = make_vehicle_bev_ego_aligned(
                ego_vehicle=ref_vehicle,
                npc_agents=npc_by_hist[hist_idx],
                out_size=self.OUT_SIZE, out_mpp=1 / self.PPM, move_px=self.move_px
            )
            c_vehicle_history.append(mask_npc)

            ped_mask = make_ped_bev_ego_aligned(
                ego_vehicle=ref_vehicle,
                ped_agents=ped_by_hist[hist_idx],
                out_size=self.OUT_SIZE, out_mpp=1 / self.PPM,
                box_px=8, move_px=self.move_px
            )
            c_walker_history.append(ped_mask)

        c_tl_history = self.get_tl_history_bev(cx5, cy5, ref_yaw)

        masks = np.stack(
            (das, path_bev, lane,
             *c_vehicle_history, *c_walker_history, *c_tl_history),
            axis=2
        )
        masks = np.transpose(masks, [2, 0, 1])
        self.bev_surfs['input'] = masks

        self.input_vis_rgb = make_input_vis_rgb(
            das, path_bev, lane,
            c_vehicle_history, c_walker_history, c_tl_history
        )
        return veh_collision, das_collision

    def render_traffic_image_from_snapshot(self, tl_snapshot):
        img = np.zeros_like(self.base_img, dtype=np.uint8)
        for lm in self.tl_landmarks:
            tl_id = int(lm.id)
            if tl_id not in self.tl_stoplines:
                continue
            info  = self.tl_stoplines[tl_id]
            state = tl_snapshot.get(tl_id, "R")
            draw_stopline_from_vertices(
                img, info["left"], info["right"], state,
                self.OFFSET_X, self.OFFSET_Y, ppm=self.PPM, thickness=6
            )
        return img

    def get_tl_history_bev(self, cx5, cy5, ref_yaw):
        """신호등 히스토리 BEV 채널 4개를 반환."""
        tl_bev_list = []
        snapshots   = self.tl_manager.get_snapshots([-1, -5, -10, -15])

        for snapshot in snapshots:
            img_traffic_hist = self.render_traffic_image_from_snapshot(snapshot)
            traffic_stop = make_local_bev_ego_aligned(
                img_traffic_hist, (cx5, cy5), ref_yaw,
                out_size=self.OUT_SIZE, out_mpp=1 / self.PPM,
                pixels_per_meter=self.PPM, fill=0, move_px=self.move_px
            )
            c_tl = np.zeros(traffic_stop.shape[:2], dtype=np.uint8)
            c_tl[np.all(traffic_stop == (0, 255, 0),   axis=2)] = 80
            c_tl[np.all(traffic_stop == (0, 255, 255), axis=2)] = 170
            c_tl[np.all(traffic_stop == (0, 0, 255),   axis=2)] = 255
            tl_bev_list.append(c_tl)

        return tl_bev_list

    # ------------------------------------------------------------------
    # 추적 오차 / 목표 속도
    # ------------------------------------------------------------------
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

        cte_mag  = float(np.linalg.norm(p - proj))
        cross_z  = float(v[0] * (p[1] - proj[1]) - v[1] * (p[0] - proj[0]))
        signed_cte = cte_mag if cross_z >= 0.0 else -cte_mag

        path_yaw    = to_rad_if_needed(self.ego_path[closest_idx][3])
        heading_error = normalize_angle_rad(path_yaw - self.ego_vehicle.yaw)
        return signed_cte, heading_error, path_yaw

    def _compute_target_speed_ms(self, closest_idx: int) -> float:
        target_v_ms = float(self.planner.compute_target_speed(
            self.ego_path, closest_idx, self.ego_vehicle,
            self.current_all_vehicles, self.ped_agents,
            self.curr_target_speed_kmh / 3.6,
            self.tl_manager,
        ))
        return float(np.clip(target_v_ms, 0.0, self.reward_cfg.max_speed_kmh / 3.6))

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 렌더링
    # ------------------------------------------------------------------
    def render(self):
        if not self.vis or self.render_mode != "human":
            return
        self._init_renderer_if_needed()

        pygame.event.pump()
        SCREEN_RES = self.SCREEN_RES
        PANEL_W    = self.PANEL_W

        v_px20, v_py20 = world_to_pixel(
            self.ego_vehicle.x, self.ego_vehicle.y,
            self.OFFSET_X, self.OFFSET_Y, 20
        )
        view_w = view_h = int(SCREEN_RES)
        cam_x  = int(clamp(v_px20 - view_w // 2, 0, self.MAP_W - view_w))
        cam_y  = int(clamp(v_py20 - view_h // 2, 0, self.MAP_H - view_h))
        camera_rect = pygame.Rect(cam_x, cam_y, view_w, view_h)

        sub_map    = self._full_map_surface.subsurface(camera_rect)
        scaled_map = pygame.transform.smoothscale(sub_map, (SCREEN_RES, SCREEN_RES))
        self._screen.blit(scaled_map, (0, 0))

        tpx20 = self.ego_path_px20[self.path_idx][0]
        tpy20 = self.ego_path_px20[self.path_idx][1]
        scale_ratio = SCREEN_RES / view_w
        pygame.draw.circle(
            self._screen, (255, 0, 0),
            (int((tpx20 - cam_x) * scale_ratio), int((tpy20 - cam_y) * scale_ratio)), 8
        )

        def to_screen(px, py):
            return int(px - cam_x), int(py - cam_y)

        for npc in self.npc_agents:
            if not npc.get("alive", True):
                continue
            veh          = npc["veh"]
            npx20, npy20 = world_to_pixel(veh.x, veh.y, self.OFFSET_X, self.OFFSET_Y, 20)
            if (cam_x - 50 <= npx20 <= cam_x + view_w + 50) and \
               (cam_y - 50 <= npy20 <= cam_y + view_h + 50):
                nsx, nsy     = to_screen(npx20, npy20)
                angle_deg    = -math.degrees(veh.yaw)
                surf         = npc.get("surf", self.npc_base)
                rotated_npc  = pygame.transform.rotozoom(surf, angle_deg, 1)
                npc_rect     = rotated_npc.get_rect(center=(nsx, nsy))
                self._screen.blit(rotated_npc, npc_rect.topleft)

        esx, esy      = to_screen(v_px20, v_py20)
        ego_angle_deg = -math.degrees(self.ego_vehicle.yaw)
        rotated_ego   = pygame.transform.rotozoom(self.car_base, ego_angle_deg, 1)
        ego_rect      = rotated_ego.get_rect(center=(esx, esy))
        self._screen.blit(rotated_ego, ego_rect.topleft)

        panel_x = SCREEN_RES
        self._screen.fill((30, 30, 30), rect=pygame.Rect(panel_x, 0, PANEL_W, SCREEN_RES))

        if hasattr(self, "input_vis_rgb"):
            blit_rgb_to_surface(self._bev_vis_surfs["input"], self.input_vis_rgb)

        TILE         = self.OUT_SIZE
        panel_blit_x = panel_x + (PANEL_W - TILE) // 2
        panel_blit_y0 = max(0, (SCREEN_RES - 5 * TILE) // 2)
        self._screen.blit(self._bev_vis_surfs["input"], (panel_blit_x, panel_blit_y0))

        active_tl_id = self.ego_path[self.path_idx][8]
        st           = self.tl_manager.state(active_tl_id)
        txet         = [st if st in ["Y", "R"] else "G"]
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
        ]
        y_offset = 15
        for line in hud_lines:
            txt_surf = self._hud_font.render(line, True, (255, 255, 0))
            self._screen.blit(txt_surf, (15, y_offset))
            y_offset += 25

        pygame.display.flip()
        self._clock.tick(self.RENDER_FPS)

    def close(self):
        if self._pygame_inited:
            pygame.quit()
            self._pygame_inited = False
