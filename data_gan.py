import os
import json
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2
import carla

from sim.traffic import *
from sim.utility import *
from sim.routeplanner import *
from sim.driving import *

from tqdm import tqdm
import multiprocessing as mp


TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town07", "Town10HD"]

PAST_FRAMES = 20
FUTURE_FRAMES = 40
SKIP_START_FRAMES = 80
SKIP_END_FRAMES = 120

BEV_SIZE = 200

MAP_MPP = 0.40
MAP_PPM = 1.0 / MAP_MPP

BEV_MPP = MAP_MPP
BEV_PPM = 1.0 / BEV_MPP

END_DISTANCE_M = 5.0
Z_TOL_M = 3.0
INITIAL_SPEED_MPS = 0.0

SIM_HZ = 20.0
DATA_HZ = 20.0
SIM_DT = 1.0 / SIM_HZ
SIM_STEPS_PER_DATA_FRAME = int(round(SIM_HZ / DATA_HZ))

CRUISE_SPEED = 12.0

PID_KP = 0.8
PID_KI = 0.2
PID_KD = 0.05

LOOKAHEAD_SMOOTH_STEP = 0.5

APPLY_BEV_X_FLIP_FOR_LABELS = True

COLORS_BGR = {
    "das": (50, 50, 50),
    "lane": (255, 255, 255),
    "stop": (0, 0, 255),
    "veh": (255, 0, 0),
    "traj": (255, 255, 255),
    "ped": (255, 255, 255),
    "route": (255, 0, 0),
}


@dataclass
class TownAssets:
    town: str
    offset_x: float
    offset_y: float
    world_map: carla.Map
    tl_landmarks: list
    drive_area_mask: np.ndarray
    drive_area_mask_high: np.ndarray
    lane_mask: np.ndarray
    lane_mask_high: np.ndarray
    route_canvas_mask: np.ndarray
    stopline_mask: np.ndarray
    z_level: int


def ensure_dirs(out_root: str):
    for layer in ["das", "lane", "stop", "veh", "traj", "ped", "route", "combined", "info"]:
        os.makedirs(os.path.join(out_root, layer), exist_ok=True)


def ped_agent_to_state(ped, pid: int) -> Dict[str, Any]:
    x, y, z = 0.0, 0.0, 0.0

    if hasattr(ped, "x") and hasattr(ped, "y"):
        x = float(ped.x)
        y = float(ped.y)
        z = float(getattr(ped, "z", 0.0))
    elif hasattr(ped, "pos"):
        x = float(ped.pos[0])
        y = float(ped.pos[1])
        z = float(ped.pos[2]) if len(ped.pos) > 2 else 0.0
    elif hasattr(ped, "location"):
        x = float(ped.location.x)
        y = float(ped.location.y)
        z = float(getattr(ped.location, "z", 0.0))

    if hasattr(ped, "yaw"):
        yaw = float(ped.yaw)
    elif hasattr(ped, "theta"):
        yaw = float(ped.theta)
    elif hasattr(ped, "heading"):
        yaw = float(ped.heading)
    else:
        yaw = 0.0

    if hasattr(ped, "v"):
        v = float(ped.v)
    elif hasattr(ped, "speed"):
        v = float(ped.speed)
    else:
        v = 0.0

    return {
        "id": int(pid),
        "x": x,
        "y": y,
        "z": z,
        "yaw": yaw,
        "v": v,
    }


def make_ped_bev_ego_aligned_from_states(
    ped_states: List[Dict[str, Any]],
    base_x: float,
    base_y: float,
    base_yaw: float,
    out_size: int = BEV_SIZE,
    ppm: float = BEV_PPM,
    ped_value: int = 180,
    box_px: int = 8,
) -> np.ndarray:
    mask = np.zeros((out_size, out_size), dtype=np.uint8)
    half = max(1, int(box_px // 2))

    for ps in ped_states:
        bx, by = world_to_ego_bev(
            float(ps["x"]),
            float(ps["y"]),
            base_x,
            base_y,
            base_yaw,
            out_size,
            ppm,
        )
        bx, by = ego_bev_xy_for_label(bx, by, out_size)

        if 0 <= bx < out_size and 0 <= by < out_size:
            x1 = max(0, int(bx) - half)
            y1 = max(0, int(by) - half)
            x2 = min(out_size - 1, int(bx) + half)
            y2 = min(out_size - 1, int(by) + half)
            cv2.rectangle(mask, (x1, y1), (x2, y2), int(ped_value), thickness=-1)

    return mask


def actor_list_to_bev_entries(
    actor_states: List[Dict[str, Any]],
    base_x: float,
    base_y: float,
    base_yaw: float,
    out_size: int = BEV_SIZE,
    ppm: float = BEV_PPM,
    actor_type: str = "npc",
) -> List[Dict[str, Any]]:
    entries = []

    for st in actor_states:
        bx, by = world_to_ego_bev(
            float(st["x"]),
            float(st["y"]),
            base_x,
            base_y,
            base_yaw,
            out_size,
            ppm,
        )
        bx2, by2 = ego_bev_xy_for_label(bx, by, out_size)
        in_bev = (0 <= bx2 < out_size) and (0 <= by2 < out_size)
        z_val = float(st.get("z", 0.0))

        if actor_type == "npc":
            entries.append(
                {
                    "id": int(st.get("id", -1)),
                    "bev": {"x": int(bx2), "y": int(by2), "in_bev": bool(in_bev)},
                    "global": {
                        "x": float(st["x"]),
                        "y": float(st["y"]),
                        "z": z_val,
                        "yaw": float(st.get("yaw", 0.0)),
                        "v": float(st.get("v", 0.0)),
                        "road_option": st.get("road_option", {"value": None, "name": None}),
                    },
                }
            )
        else:
            entries.append(
                {
                    "id": int(st.get("id", -1)),
                    "bev": {"x": int(bx2), "y": int(by2), "in_bev": bool(in_bev)},
                    "global": {
                        "x": float(st["x"]),
                        "y": float(st["y"]),
                        "z": z_val,
                        "yaw": float(st.get("yaw", 0.0)),
                        "v": float(st.get("v", 0.0)),
                    },
                }
            )

    return entries


def load_town_assets(town: str) -> TownAssets:
    offset_dir = f"sim_using_data/data20/{town}"
    data_dir = f"sim_using_data/data2/{town}"
    xodr_path = f"sim_using_data/Town/{town}.xodr"

    offset_path = f"{offset_dir}/world_offset.npy"
    if not os.path.exists(offset_path):
        raise FileNotFoundError(f"Offset file not found: {offset_path}")

    world_offset = np.load(offset_path)
    offset_x, offset_y = float(world_offset[0]), float(world_offset[1])

    drive_area_mask = cv2.imread(f"{data_dir}/das_full.png", cv2.IMREAD_GRAYSCALE)
    lane_mask = cv2.imread(f"{data_dir}/lane_full.png", cv2.IMREAD_GRAYSCALE)

    if town in ["Town04", "Town05"]:
        drive_area_mask_high = cv2.imread(f"{data_dir}/das_full_high.png", cv2.IMREAD_GRAYSCALE)
        lane_mask_high = cv2.imread(f"{data_dir}/lane_full_high.png", cv2.IMREAD_GRAYSCALE)
    else:
        drive_area_mask_high = cv2.imread(f"{data_dir}/das_full.png", cv2.IMREAD_GRAYSCALE)
        lane_mask_high = cv2.imread(f"{data_dir}/lane_full.png", cv2.IMREAD_GRAYSCALE)

    if town == "Town04":
        z_level = 3
    elif town == "Town05":
        z_level = 7
    else:
        z_level = 0

    route_canvas_mask = cv2.imread(f"{data_dir}/{town}.png", cv2.IMREAD_GRAYSCALE)
    stopline_mask = cv2.imread(f"{data_dir}/stoplines.png", cv2.IMREAD_GRAYSCALE)

    if drive_area_mask is None or lane_mask is None or route_canvas_mask is None or stopline_mask is None:
        raise FileNotFoundError(f"Missing one or more raster files for {town} under {data_dir}")

    with open(xodr_path, "r", encoding="utf-8") as f:
        xodr_content = f.read()

    world_map = carla.Map(town, xodr_content)
    landmarks = world_map.get_all_landmarks()
    tl_landmarks = [lm for lm in landmarks if is_traffic_light_landmark(lm)]

    return TownAssets(
        town=town,
        offset_x=offset_x,
        offset_y=offset_y,
        world_map=world_map,
        tl_landmarks=tl_landmarks,
        drive_area_mask=drive_area_mask,
        drive_area_mask_high=drive_area_mask_high,
        lane_mask=lane_mask,
        lane_mask_high=lane_mask_high,
        route_canvas_mask=route_canvas_mask,
        stopline_mask=stopline_mask,
        z_level=z_level,
    )


def alpha_blend_mask(img, mask_u8, color_bgr, alpha=0.4):
    if mask_u8 is None:
        return

    m = mask_u8 > 0
    if not np.any(m):
        return

    color = np.array(color_bgr, dtype=np.float32)
    base = img[m].astype(np.float32)
    blended = base * (1.0 - alpha) + color * alpha
    img[m] = blended.astype(np.uint8)


def make_combined_bgr(
    das_200,
    lane_200,
    stop_200,
    veh_200,
    route_200,
    ped_200=None,
    traj_200=None,
    traj_alpha=0.5,
    route_alpha=0.35,
) -> np.ndarray:
    h, w = das_200.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)

    def paint(mask, color_bgr):
        if mask is None:
            return
        m = mask > 0
        img[m] = color_bgr

    paint(das_200, COLORS_BGR["das"])
    paint(lane_200, COLORS_BGR["lane"])
    paint(stop_200, COLORS_BGR["stop"])
    paint(veh_200, COLORS_BGR["veh"])
    paint(ped_200, COLORS_BGR["ped"])

    alpha_blend_mask(img, route_200, COLORS_BGR["route"], alpha=route_alpha)

    if traj_200 is not None:
        mask = traj_200 > 0
        if np.any(mask):
            color = np.array(COLORS_BGR["traj"], dtype=np.uint8)
            overlay = img[mask].astype(float)
            blended = (overlay * (1 - traj_alpha) + color * traj_alpha).astype(np.uint8)
            img[mask] = blended

    return img


def build_nav_points(
    ego_path,
    center_path_idx,
    base_x,
    base_y,
    base_yaw,
    bev_size,
    ppm,
    step=5,
    span=20,
):
    nav_points = []
    if not ego_path:
        return nav_points

    offsets = list(range(-span, span + 1, step))
    max_i = len(ego_path) - 1

    for d in offsets:
        pi = center_path_idx + d
        pi = 0 if pi < 0 else (max_i if pi > max_i else pi)

        gx, gy = float(ego_path[pi][0]), float(ego_path[pi][1])

        bx, by = world_to_ego_bev(
            gx,
            gy,
            base_x,
            base_y,
            base_yaw,
            bev_size,
            ppm,
        )
        bx2, by2 = ego_bev_xy_for_label(bx, by, bev_size)
        in_bev = (0 <= bx2 < bev_size) and (0 <= by2 < bev_size)

        nav_points.append(
            {
                "d_path_idx": int(d),
                "path_idx": int(pi),
                "global": {"x": gx, "y": gy},
                "bev": {"x": int(bx2), "y": int(by2), "in_bev": bool(in_bev)},
            }
        )

    return nav_points


def road_option_to_dict(ro) -> Dict[str, Any]:
    try:
        return {"value": int(ro.value), "name": str(ro.name)}
    except Exception:
        return {"value": int(ro), "name": None}


def tl_state_to_str(st) -> Optional[str]:
    if st is None:
        return None
    m = {"G": "GREEN", "Y": "YELLOW", "R": "RED"}
    return m.get(st, str(st))


def ego_bev_xy_for_label(px: int, py: int, size: int) -> Tuple[int, int]:
    if APPLY_BEV_X_FLIP_FOR_LABELS:
        return size - 1 - int(px), int(py)
    return int(px), int(py)


def build_temp_agents_for_vehicle_mask(
    ego_state: Dict[str, Any],
    npc_states: List[Dict[str, Any]],
) -> Tuple[Any, List[Dict[str, Any]]]:
    ego_tmp = BicycleModel(
        float(ego_state["x"]),
        float(ego_state["y"]),
        float(ego_state["z"]),
        float(ego_state["yaw"]),
    )
    ego_tmp.v = float(ego_state.get("v", 0.0))

    npc_agents_tmp = []
    for ns in npc_states:
        v = BicycleModel(
            float(ns["x"]),
            float(ns["y"]),
            float(ns["z"]),
            float(ns["yaw"]),
        )
        v.v = float(ns.get("v", 0.0))
        npc_agents_tmp.append({"veh": v, "alive": True})

    return ego_tmp, npc_agents_tmp


def simulate_episode(
    assets: TownAssets,
    num_npcs: int,
    max_steps_data_frames: int,
    seed: int,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    ped_json = f"sim_using_data/Town/{assets.town}_ped_graph.json"
    ped_graph = PedGraph(ped_json, grid=2.0)

    world_map = assets.world_map

    tl_manager = TrafficLight_Manager(
        world_map,
        assets.tl_landmarks,
        t_green=10.0,
        t_yellow=3.0,
        t_all_red=2.0,
    )

    ego_results = generate_random_routes(
        world_map,
        assets.drive_area_mask,
        assets.offset_x,
        assets.offset_y,
        num_routes=1,
        sample_dist=4000.0,
        max_len=12000,
        ppm=MAP_PPM,
        invert_y=False,
    )
    ego_path = ego_results[0] if ego_results else []
    if len(ego_path) < 5:
        return {"records": [], "ego_path": []}

    npc_routes = generate_random_routes(
        world_map,
        assets.drive_area_mask,
        assets.offset_x,
        assets.offset_y,
        num_routes=num_npcs,
        sample_dist=4000.0,
        max_len=8000,
        ppm=MAP_PPM,
        invert_y=False,
    )

    ego_vehicle = BicycleModel(ego_path[0][0], ego_path[0][1], ego_path[0][2], ego_path[0][3])
    ego_vehicle.v = float(INITIAL_SPEED_MPS)

    ped_agents = []
    for _ in range(40):
        try:
            p = PedAgent(
                ped_graph,
                speed=np.random.uniform(1.0, 1.8),
                reach=0.7,
                max_yaw_rate=8.0,
                jaywalk_cost=0.7,
            )
            if not p.alive:
                continue
            ped_agents.append(p)
        except Exception:
            continue

    planner = LongitudinalPlanner()
    speed_pid = PID(PID_KP, PID_KI, PID_KD, out_min=-1.0, out_max=1.0)

    npc_agents = []
    for k, route in enumerate(npc_routes, start=1):
        if len(route) < 2:
            continue

        npc_vehicle = BicycleModel(route[0][0], route[0][1], route[0][2], route[0][3])
        npc_pid = PID(PID_KP, PID_KI, PID_KD, out_min=-1.0, out_max=1.0)
        cruise_k = CRUISE_SPEED * np.random.uniform(0.7, 1.1)
        npc_planner = NPCLongitudinalPlanner()

        npc_agents.append(
            {
                "id": k,
                "path": route,
                "veh": npc_vehicle,
                "idx": 0,
                "pid": npc_pid,
                "planner": npc_planner,
                "cruise": float(cruise_k),
                "alive": True,
                "lookahead": 10.0,
                "last_steer": 0.0,
            }
        )

    sim_time = 0.0
    path_idx = 0
    closest_idx = 0
    current_lookahead = 10.0

    throttle_cmd = 0.0
    brake_cmd = 0.0
    delta_clamped = 0.0
    steer_norm = 0.0
    target_v = 0.0

    goal_x, goal_y = float(ego_path[-1][0]), float(ego_path[-1][1])

    records: List[Dict[str, Any]] = []
    dt = SIM_DT

    for frame_i in tqdm(range(max_steps_data_frames), desc="episode frames", unit="frame"):
        reached_route_tail = False

        for _ in range(SIM_STEPS_PER_DATA_FRAME):
            sim_time += dt
            tl_manager.step(dt)

            current_all_vehicles = [{"id": 0, "veh": ego_vehicle}]
            for npc in npc_agents:
                if npc["alive"]:
                    current_all_vehicles.append({"id": npc["id"], "veh": npc["veh"]})

            try:
                ro_val = ego_path[path_idx][9].value
            except Exception:
                ro_val = int(ego_path[path_idx][9]) if len(ego_path[path_idx]) > 9 else 0

            if ro_val in [1, 2]:
                target_lookahead = 4.0
            else:
                tx_next, ty_next = ego_path[path_idx][0], ego_path[path_idx][1]
                target_heading = math.atan2(ty_next - ego_vehicle.y, tx_next - ego_vehicle.x)
                heading_err = abs(normalize_angle_rad(target_heading - ego_vehicle.yaw))
                target_lookahead = 4.0 if heading_err > 0.01 else 10.0

            if current_lookahead < target_lookahead:
                current_lookahead = min(target_lookahead, current_lookahead + LOOKAHEAD_SMOOTH_STEP)
            elif current_lookahead > target_lookahead:
                current_lookahead = max(target_lookahead, current_lookahead - LOOKAHEAD_SMOOTH_STEP)

            path_idx = find_lookahead_index(
                ego_path,
                path_idx,
                ego_vehicle.x,
                ego_vehicle.y,
                current_lookahead,
            )

            tail_threshold = len(ego_path) - int(current_lookahead + 3)
            if path_idx >= tail_threshold:
                reached_route_tail = True
                break

            tx, ty = ego_path[path_idx][0], ego_path[path_idx][1]
            delta = pure_pursuit_control(ego_vehicle, tx, ty, current_lookahead)
            delta_clamped = clamp(delta, -ego_vehicle.max_steer, ego_vehicle.max_steer)
            steer_norm = clamp(-delta_clamped / ego_vehicle.max_steer, -1.0, 1.0)

            closest_idx = find_closest_index(ego_path, ego_vehicle.x, ego_vehicle.y)
            target_v = planner.compute_target_speed(
                ego_path,
                closest_idx,
                ego_vehicle,
                current_all_vehicles,
                ped_agents,
                CRUISE_SPEED,
                tl_manager,
            )

            if target_v < 0.1 or getattr(planner, "is_waiting", False):
                throttle_cmd = 0.0
                brake_cmd = 1.0
                speed_pid.reset()
            else:
                err_v = target_v - ego_vehicle.v
                u = speed_pid.step(err_v, dt)
                if u >= 0:
                    throttle_cmd = clamp(u, 0.0, 1.0)
                    brake_cmd = 0.0
                else:
                    throttle_cmd = 0.0
                    brake_cmd = clamp(-u, 0.0, 1.0)

            ego_vehicle.update(delta_clamped, throttle_cmd, brake_cmd, dt)

            for npc in npc_agents:
                if not npc["alive"]:
                    continue

                veh = npc["veh"]
                path = npc["path"]
                idx = npc["idx"]
                pid = npc["pid"]
                planner_npc = npc["planner"]
                cruise_k = npc["cruise"]
                lookahead_n = npc["lookahead"]

                try:
                    n_ro_val = path[idx][9].value
                except Exception:
                    n_ro_val = int(path[idx][9]) if len(path[idx]) > 9 else 0

                if n_ro_val in [1, 2]:
                    n_target_lookahead = 4.0
                else:
                    tx_next, ty_next = path[idx][0], path[idx][1]
                    target_heading = math.atan2(ty_next - veh.y, tx_next - veh.x)
                    heading_err = abs(normalize_angle_rad(target_heading - veh.yaw))
                    n_target_lookahead = 4.0 if heading_err > 0.01 else 10.0

                if lookahead_n < n_target_lookahead:
                    lookahead_n = min(n_target_lookahead, lookahead_n + LOOKAHEAD_SMOOTH_STEP)
                elif lookahead_n > n_target_lookahead:
                    lookahead_n = max(n_target_lookahead, lookahead_n - LOOKAHEAD_SMOOTH_STEP)

                idx = find_lookahead_index(path, idx, veh.x, veh.y, lookahead_n)

                if idx >= len(path) - int(lookahead_n + 3):
                    start_point = carla.Location(x=float(veh.x), y=float(veh.y), z=0.0)
                    new_routes = generate_random_routes_roop(
                        world_map,
                        start_point,
                        num_routes=1,
                        sample_dist=1000.0,
                        max_len=4000,
                    )
                    if new_routes and len(new_routes[0]) > 3:
                        npc["path"] = new_routes[0]
                        path = npc["path"]
                        idx = 0
                        veh.x, veh.y, veh.yaw = path[0][0], path[0][1], path[0][3]
                        veh.v = float(INITIAL_SPEED_MPS)
                        pid.reset()
                    else:
                        npc["alive"] = False
                        continue

                ntx, nty = path[idx][0], path[idx][1]
                ndelta = pure_pursuit_control(veh, ntx, nty, lookahead_n)
                ndelta_clamped = clamp(ndelta, -veh.max_steer, veh.max_steer)

                n_closest = find_closest_index(path, veh.x, veh.y)
                ntarget_v = planner_npc.compute_target_speed(
                    path,
                    n_closest,
                    veh,
                    current_all_vehicles,
                    ped_agents,
                    cruise_k,
                    tl_manager,
                )

                if ntarget_v < 0.1 or getattr(planner_npc, "is_waiting", False):
                    nthrottle = 0.0
                    nbrake = 1.0
                    pid.reset()
                else:
                    nerr_v = ntarget_v - veh.v
                    nu = pid.step(nerr_v, dt)
                    if nu >= 0:
                        nthrottle = clamp(nu, 0.0, 1.0)
                        nbrake = 0.0
                    else:
                        nthrottle = 0.0
                        nbrake = clamp(-nu, 0.0, 1.0)

                veh.update(ndelta_clamped, nthrottle, nbrake, dt)

                npc["idx"] = idx
                npc["last_steer"] = float(ndelta_clamped)
                npc["lookahead"] = float(lookahead_n)

            for ped in ped_agents:
                ped.step(dt)

        if path_idx >= len(ego_path):
            path_idx = len(ego_path) - 1
        if path_idx < 0:
            path_idx = 0

        current_tl_id = int(ego_path[path_idx][8]) if len(ego_path[path_idx]) > 8 else -1
        st = tl_manager.state(current_tl_id) if current_tl_id != -1 else None
        ro = ego_path[path_idx][9] if len(ego_path[path_idx]) > 9 else 0

        ego_vehicle.z = ego_path[path_idx][2]
        for npc in npc_agents:
            if npc["alive"]:
                npc["veh"].z = npc["path"][npc["idx"]][2]

        npc_states = []
        for npc in npc_agents:
            if not npc["alive"]:
                continue

            v = npc["veh"]
            try:
                nro = npc["path"][npc["idx"]][9]
            except Exception:
                nro = 0

            npc_states.append(
                {
                    "id": int(npc["id"]),
                    "x": float(v.x),
                    "y": float(v.y),
                    "z": float(v.z),
                    "yaw": float(v.yaw),
                    "v": float(getattr(v, "v", 0.0)),
                    "road_option": road_option_to_dict(nro),
                    "delta": float(npc.get("last_steer", 0.0)),
                    "L": float(getattr(v, "L", 2.875)),
                }
            )

        ped_states = []
        for pid, ped in enumerate(ped_agents):
            ped_states.append(ped_agent_to_state(ped, pid))

        records.append(
            {
                "frame": int(frame_i),
                "sim_time": float(sim_time),
                "ego": {
                    "x": float(ego_vehicle.x),
                    "y": float(ego_vehicle.y),
                    "z": float(ego_vehicle.z),
                    "yaw": float(ego_vehicle.yaw),
                    "v": float(getattr(ego_vehicle, "v", 0.0)),
                    "steer": float(steer_norm),
                    "delta": float(delta_clamped),
                    "throttle": float(throttle_cmd),
                    "brake": float(brake_cmd),
                    "gear": int(getattr(ego_vehicle, "gear", 0)),
                    "target_x": float(ego_path[closest_idx][0]),
                    "target_y": float(ego_path[closest_idx][1]),
                    "target_yaw": float(ego_path[closest_idx][3]),
                },
                "path_idx": int(path_idx),
                "road_option": road_option_to_dict(ro),
                "traffic_light": {
                    "id": int(current_tl_id),
                    "state": tl_state_to_str(st),
                    "raw": None if st is None else str(st),
                },
                "planner_is": bool(planner.traffic),
                "npcs": npc_states,
                "peds": ped_states,
                "target_speed": float(target_v),
            }
        )

        if reached_route_tail:
            break

        dist_to_goal = math.hypot(ego_vehicle.x - goal_x, ego_vehicle.y - goal_y)
        if dist_to_goal <= END_DISTANCE_M:
            break
        if path_idx >= len(ego_path) - 5:
            break

    return {"records": records, "ego_path": ego_path}


def render_route_global_mask_u8(
    ego_path,
    assets,
    value=255,
    thickness_bev_px=20,
    base_z: Optional[float] = None,
    z_tol: float = Z_TOL_M,
):
    h, w = assets.route_canvas_mask.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not ego_path or len(ego_path) < 2:
        return mask

    thickness_global_px = max(1, int(round(thickness_bev_px * (BEV_MPP / MAP_MPP))))

    segments = []
    cur = []

    for p in ego_path:
        if isinstance(p, dict):
            gx, gy = float(p["x"]), float(p["y"])
            pz = float(p.get("z", base_z if base_z is not None else 0.0))
        else:
            gx, gy = float(p[0]), float(p[1])
            pz = float(p[2]) if len(p) >= 3 else (base_z if base_z is not None else 0.0)

        if base_z is not None and abs(pz - base_z) > z_tol:
            if len(cur) >= 2:
                segments.append(cur)
            cur = []
            continue

        px, py = world_to_pixel(gx, gy, assets.offset_x, assets.offset_y,MAP_PPM)
        cur.append([int(px), int(py)])

    if len(cur) >= 2:
        segments.append(cur)

    if not segments:
        return mask

    for seg in segments:
        pts_np = np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            mask,
            [pts_np],
            isClosed=False,
            color=int(value),
            thickness=thickness_global_px,
            lineType=cv2.LINE_8,
        )

    return mask


def build_and_save_sample(
    out_root: str,
    assets: TownAssets,
    episode_id: int,
    sample_id: int,
    records: List[Dict[str, Any]],
    ego_path,
    idx: int,
    save_q=None,
    png_comp: int = 1,
):
    cur = records[idx]
    ego = cur["ego"]
    planner_is = cur["planner_is"]

    base_x = float(ego["x"])
    base_y = float(ego["y"])
    base_z = float(ego["z"])
    base_yaw = float(ego["yaw"])
    path_idx_now = int(cur.get("path_idx", 0))

    use_multi_level_filter = assets.town in {"Town04", "Town05"}

    route_global_mask_u8 = render_route_global_mask_u8(
        ego_path=ego_path,
        assets=assets,
        value=255,
        thickness_bev_px=20,
        base_z=base_z,
        z_tol=Z_TOL_M,
    )

    cx, cy = world_to_pixel(base_x, base_y, assets.offset_x, assets.offset_y,MAP_PPM)

    if base_z > assets.z_level:
        das_200 = make_local_bev_ego_aligned(
            assets.drive_area_mask_high,
            (cx, cy),
            base_yaw,
            out_size=BEV_SIZE,
            out_mpp=BEV_MPP,
            pixels_per_meter=MAP_PPM,
            fill=0,
        )
        lane_200 = make_local_bev_ego_aligned(
            assets.lane_mask_high,
            (cx, cy),
            base_yaw,
            out_size=BEV_SIZE,
            out_mpp=BEV_MPP,
            pixels_per_meter=MAP_PPM,
            fill=0,
        )
    else:
        das_200 = make_local_bev_ego_aligned(
            assets.drive_area_mask,
            (cx, cy),
            base_yaw,
            out_size=BEV_SIZE,
            out_mpp=BEV_MPP,
            pixels_per_meter=MAP_PPM,
            fill=0,
        )
        lane_200 = make_local_bev_ego_aligned(
            assets.lane_mask,
            (cx, cy),
            base_yaw,
            out_size=BEV_SIZE,
            out_mpp=BEV_MPP,
            pixels_per_meter=MAP_PPM,
            fill=0,
        )

    stop_200 = make_local_bev_ego_aligned(
        assets.stopline_mask,
        (cx, cy),
        base_yaw,
        out_size=BEV_SIZE,
        out_mpp=BEV_MPP,
        pixels_per_meter=MAP_PPM,
        fill=0,
    )

    nav_points = build_nav_points(
        ego_path=ego_path,
        center_path_idx=path_idx_now,
        base_x=base_x,
        base_y=base_y,
        base_yaw=base_yaw,
        bev_size=BEV_SIZE,
        ppm=BEV_PPM,
        step=5,
        span=20,
    )

    route_200 = make_local_bev_ego_aligned(
        route_global_mask_u8,
        (cx, cy),
        base_yaw,
        out_size=BEV_SIZE,
        out_mpp=BEV_MPP*2,
        pixels_per_meter=MAP_PPM,
        fill=0,
    )

    filtered_npcs = []
    for ns in cur.get("npcs", []):
        nz = float(ns.get("z", base_z))
        if abs(nz - base_z) > Z_TOL_M:
            continue
        ns2 = dict(ns)
        ns2["z"] = nz
        filtered_npcs.append(ns2)

    ego_with_z = dict(ego)
    ego_with_z["z"] = float(base_z)

    ego_tmp, npc_agents_tmp = build_temp_agents_for_vehicle_mask(ego_with_z, filtered_npcs)
    veh_200, _, _ = make_vehicle_bev_ego_aligned(
        ego_tmp,
        npc_agents_tmp,
        out_size=BEV_SIZE,
        out_mpp=BEV_MPP,
        z_tol_m=Z_TOL_M,
    )

    ped_states_cur = cur.get("peds", [])
    if use_multi_level_filter:
        ped_states_filtered = []
        for ps in ped_states_cur:
            pz = float(ps.get("z", base_z))
            if abs(pz - base_z) > Z_TOL_M:
                continue
            ps2 = dict(ps)
            ps2["z"] = pz
            ped_states_filtered.append(ps2)
        ped_states_cur = ped_states_filtered

    ped_200 = make_ped_bev_ego_aligned_from_states(
        ped_states_cur,
        base_x=base_x,
        base_y=base_y,
        base_yaw=base_yaw,
        out_size=BEV_SIZE,
        ppm=BEV_PPM,
        ped_value=180,
        box_px=8,
    )

    traj_200 = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.uint8)

    predict_time = 2.0
    max_dist_limit = 10.0
    step_t = 0.05

    def draw_traj_rollout(x, y, yaw, v, delta, L, intensity, radius=2, z=None, z_tol=Z_TOL_M):
        if z is not None and abs(float(z) - float(base_z)) > z_tol:
            return
        if v <= 0.5:
            return

        accumulated = 0.0
        tx, ty, tyaw = float(x), float(y), float(yaw)

        for _ in range(int(predict_time / step_t)):
            d = float(v) * step_t
            accumulated += d
            if accumulated > max_dist_limit:
                break

            tx += math.cos(tyaw) * d
            ty += math.sin(tyaw) * d
            tyaw += (float(v) / float(L)) * math.tan(float(delta)) * step_t

            bx, by = world_to_ego_bev(
                tx,
                ty,
                base_x,
                base_y,
                base_yaw,
                BEV_SIZE,
                BEV_PPM,
            )
            bx, by = ego_bev_xy_for_label(bx, by, BEV_SIZE)

            if 0 <= bx < BEV_SIZE and 0 <= by < BEV_SIZE:
                cv2.circle(traj_200, (int(bx), int(by)), int(radius), int(intensity), -1)

    ego_L = float(getattr(ego_tmp, "L", 2.875))
    draw_traj_rollout(
        x=ego["x"],
        y=ego["y"],
        yaw=ego["yaw"],
        v=ego.get("v", 0.0),
        delta=ego.get("delta", 0.0),
        L=ego_L,
        intensity=100,
        radius=5,
        z=base_z,
    )

    for ns in filtered_npcs:
        draw_traj_rollout(
            x=ns["x"],
            y=ns["y"],
            yaw=ns["yaw"],
            v=ns.get("v", 0.0),
            delta=ns.get("delta", 0.0),
            L=float(ns.get("L", 2.5)),
            intensity=200,
            radius=5,
            z=ns.get("z", base_z),
        )

    combined = make_combined_bgr(
        das_200,
        lane_200,
        stop_200,
        veh_200,
        route_200=route_200,
        ped_200=ped_200,
        traj_200=traj_200,
        traj_alpha=0.5,
    )

    past_entries = []
    for j in range(idx - PAST_FRAMES, idx):
        rj = records[j]
        ej = rj["ego"]

        bx, by = world_to_ego_bev(
            float(ej["x"]),
            float(ej["y"]),
            base_x,
            base_y,
            base_yaw,
            BEV_SIZE,
            BEV_PPM,
        )
        bx2, by2 = ego_bev_xy_for_label(bx, by, BEV_SIZE)
        in_bev = (0 <= bx2 < BEV_SIZE) and (0 <= by2 < BEV_SIZE)

        past_entries.append(
            {
                "dt_frames": int(j - idx),
                "bev": {"x": int(bx2), "y": int(by2), "in_bev": bool(in_bev)},
                "global": {
                    "x": float(ej["x"]),
                    "y": float(ej["y"]),
                    "z": float(ej.get("z", base_z)),
                    "yaw": float(ej["yaw"]),
                    "road_option": rj["road_option"],
                },
                "state": {
                    "v": float(ej.get("v", 0.0)),
                    "steer": float(ej.get("steer", 0.0)),
                    "throttle": float(ej.get("throttle", 0.0)),
                    "brake": float(ej.get("brake", 0.0)),
                    "gear": int(ej.get("gear", 0)),
                },
            }
        )

    future_entries = []
    for j in range(idx + 1, idx + FUTURE_FRAMES + 1):
        rj = records[j]
        ej = rj["ego"]

        bx, by = world_to_ego_bev(
            float(ej["x"]),
            float(ej["y"]),
            base_x,
            base_y,
            base_yaw,
            BEV_SIZE,
            BEV_PPM,
        )
        bx2, by2 = ego_bev_xy_for_label(bx, by, BEV_SIZE)
        in_bev = (0 <= bx2 < BEV_SIZE) and (0 <= by2 < BEV_SIZE)

        future_entries.append(
            {
                "dt_frames": int(j - idx),
                "bev": {"x": int(bx2), "y": int(by2), "in_bev": bool(in_bev)},
                "global": {
                    "x": float(ej["x"]),
                    "y": float(ej["y"]),
                    "z": float(ej.get("z", base_z)),
                    "yaw": float(ej["yaw"]),
                    "road_option": rj["road_option"],
                },
                "state": {
                    "v": float(ej.get("v", 0.0)),
                    "steer": float(ej.get("steer", 0.0)),
                    "throttle": float(ej.get("throttle", 0.0)),
                    "brake": float(ej.get("brake", 0.0)),
                    "gear": int(ej.get("gear", 0)),
                },
            }
        )

    def filter_actors_by_level(actor_states):
        out = []
        for st in actor_states:
            az = float(st.get("z", base_z))
            if abs(az - base_z) > Z_TOL_M:
                continue
            st2 = dict(st)
            st2["z"] = az
            out.append(st2)
        return out

    npc_entries_cur = actor_list_to_bev_entries(
        filtered_npcs,
        base_x=base_x,
        base_y=base_y,
        base_yaw=base_yaw,
        out_size=BEV_SIZE,
        ppm=BEV_PPM,
        actor_type="npc",
    )

    ped_entries_cur = actor_list_to_bev_entries(
        ped_states_cur,
        base_x=base_x,
        base_y=base_y,
        base_yaw=base_yaw,
        out_size=BEV_SIZE,
        ppm=BEV_PPM,
        actor_type="ped",
    )

    info = {
        "town": assets.town,
        "episode_id": int(episode_id),
        "sample_id": int(sample_id),
        "source_frame_index": int(idx),
        "timestamp": time.time(),
        "bev_spec": {
            "size": int(BEV_SIZE),
            "ppm": float(BEV_PPM),
            "mpp": float(BEV_MPP),
            "x_flip_labels": bool(APPLY_BEV_X_FLIP_FOR_LABELS),
            "origin": "ego_center",
            "alignment": "ego_yaw_aligned",
        },
        "world_offset": {"x": float(assets.offset_x), "y": float(assets.offset_y)},
        "current": {
            "sim_time": float(cur["sim_time"]),
            "ego_global": {
                "x": float(ego["x"]),
                "y": float(ego["y"]),
                "z": float(base_z),
                "yaw": float(ego["yaw"]),
                "road_option": cur["road_option"],
            },
            "ego_state": {
                "v": float(ego.get("v", 0.0)),
                "steer": float(ego.get("steer", 0.0)),
                "delta": float(ego.get("delta", 0.0)),
                "throttle": float(ego.get("throttle", 0.0)),
                "brake": float(ego.get("brake", 0.0)),
                "gear": int(ego.get("gear", 0)),
                "target_speed": float(cur.get("target_speed", 0.0)),
            },
            "target": {
                "target_x": float(ego.get("target_x", 0.0)),
                "target_y": float(ego.get("target_y", 0.0)),
                "target_yaw": float(ego.get("target_yaw", 0.0)),
            },
            "traffic_light": cur.get("traffic_light", {}),
            "npcs": npc_entries_cur,
            "peds": ped_entries_cur,
            "nav_points": nav_points,
        },
        "planner_is": planner_is,
        "past": past_entries,
        "future": future_entries,
    }

    stem = f"{sample_id:08d}"

    imgs_dict = {
        "route": route_200,
        "das": das_200,
        "lane": lane_200,
        "stop": stop_200,
        "veh": veh_200,
        "ped": ped_200,
        "traj": traj_200,
        "combined": combined,
    }

    if save_q is not None:
        save_q.put((out_root, stem, imgs_dict, info))
    else:
        png_params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_comp)]
        for subdir, img in imgs_dict.items():
            cv2.imwrite(os.path.join(out_root, subdir, f"{stem}.png"), img, png_params)

        with open(os.path.join(out_root, "info", f"{stem}.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False)


def get_npc_count_by_town(town_name):
    town_config = {
        "Town01": 25,
        "Town02": 25,
        "Town03": 50,
        "Town07": 40,
        "Town04": 100,
        "Town06": 90,
        "Town05": 120,
        "Town10HD": 70,
    }

    for key, value in town_config.items():
        if key in town_name:
            return value

    return 40


def writer_loop(q: mp.Queue, png_compression: int = 1):
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]

    while True:
        item = q.get()
        if item is None:
            break

        out_root, stem, imgs_dict, info = item

        for subdir, img in imgs_dict.items():
            out_path = os.path.join(out_root, subdir, f"{stem}.png")
            cv2.imwrite(out_path, img, png_params)

        info_path = os.path.join(out_root, "info", f"{stem}.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False)


def town_seed_offset(name: str) -> int:
    return sum(ord(c) for c in name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data", help="output root dir")
    parser.add_argument("--target_samples", type=int, default=20000, help="total samples to generate")
    parser.add_argument("--max_episode_frames", type=int, default=2000, help="max data frames per episode")
    parser.add_argument("--seed", type=int, default=0, help="base random seed")
    parser.add_argument("--towns", type=str, default=",".join(TOWNS), help="comma-separated town list")
    parser.add_argument("--async_save", action="store_true", help="use async writer process")
    parser.add_argument("--save_queue", type=int, default=256, help="max queue size for async saving")
    parser.add_argument("--png_comp", type=int, default=1, help="png compression 0~9")
    args = parser.parse_args()

    out_root = args.out
    os.makedirs(out_root, exist_ok=True)

    towns = [t.strip() for t in args.towns.split(",") if t.strip()]
    if not towns:
        raise ValueError("No towns provided")

    save_q = None
    writer_proc = None

    if args.async_save:
        save_q = mp.Queue(maxsize=int(args.save_queue))
        writer_proc = mp.Process(
            target=writer_loop,
            args=(save_q, int(args.png_comp)),
            daemon=True,
        )
        writer_proc.start()

    cache = {}
    episode_id = 0
    total_samples_written = 0

    for town in towns:
        town_root = os.path.join(out_root, town)
        ensure_dirs(town_root)

        if town not in cache:
            cache[town] = load_town_assets(town)
        assets = cache[town]

        dynamic_npc_count = get_npc_count_by_town(town)
        print(f"Town: {town}, Assigned NPCs: {dynamic_npc_count}")

        sample_id = 0

        while sample_id < args.target_samples:
            ep_seed = args.seed + episode_id * 1000 + town_seed_offset(town)

            result = simulate_episode(
                assets=assets,
                num_npcs=dynamic_npc_count,
                max_steps_data_frames=args.max_episode_frames,
                seed=ep_seed,
            )
            episode_id += 1

            records = result["records"]
            ego_path = result["ego_path"]

            min_required_len = SKIP_START_FRAMES + SKIP_END_FRAMES + PAST_FRAMES + FUTURE_FRAMES + 5
            if len(records) < min_required_len:
                continue

            start_idx = max(SKIP_START_FRAMES, PAST_FRAMES)
            end_idx = min(len(records) - 1 - SKIP_END_FRAMES, len(records) - 1 - FUTURE_FRAMES)
            if end_idx <= start_idx:
                continue

            for idx in range(start_idx, end_idx + 1):
                if sample_id >= args.target_samples:
                    break

                build_and_save_sample(
                    out_root=town_root,
                    assets=assets,
                    episode_id=episode_id,
                    sample_id=sample_id,
                    records=records,
                    ego_path=ego_path,
                    idx=idx,
                    save_q=save_q,
                    png_comp=args.png_comp,
                )

                sample_id += 1
                total_samples_written += 1

        print(f"[{town}] Done: {sample_id} samples into {town_root}/")

    if save_q is not None and writer_proc is not None:
        save_q.put(None)
        writer_proc.join()

    print(f"Done. Generated {total_samples_written} samples into {out_root}/")


if __name__ == "__main__":
    main()