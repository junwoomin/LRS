import carla
import random
from collections import defaultdict, deque


import json
import os
def _ang_diff(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def get_junction_id_for_landmark(world_map, lm, lane_type=carla.LaneType.Driving,
                                 search_dist=30.0, step=1.0):
    loc = lm.transform.location
    wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=lane_type)
    if wp is None:
        return None

    if wp.is_junction:
        j = wp.get_junction()
        return int(j.id) if j is not None else None

    dist = 0.0
    cur = wp
    while dist < search_dist:
        nxt = cur.next(step)
        if not nxt:
            break
        cur = nxt[0]
        dist += step
        if cur.is_junction:
            j = cur.get_junction()
            return int(j.id) if j is not None else None

    dist = 0.0
    cur = wp
    while dist < search_dist:
        prv = cur.previous(step)
        if not prv:
            break
        cur = prv[0]
        dist += step
        if cur.is_junction:
            j = cur.get_junction()
            return int(j.id) if j is not None else None

    return None


def split_into_approaches(lms, gap_deg=45.0):
    if not lms:
        return []

    lms_sorted = sorted(lms, key=lambda x: float(x.transform.rotation.yaw) % 360.0)

    groups = []
    current_group = [lms_sorted[0]]

    for i in range(1, len(lms_sorted)):
        prev_yaw = float(lms_sorted[i - 1].transform.rotation.yaw) % 360.0
        curr_yaw = float(lms_sorted[i].transform.rotation.yaw) % 360.0

        if _ang_diff(curr_yaw, prev_yaw) < gap_deg:
            current_group.append(lms_sorted[i])
        else:
            groups.append(current_group)
            current_group = [lms_sorted[i]]

    if current_group:
        first_yaw = float(lms_sorted[0].transform.rotation.yaw) % 360.0
        last_yaw = float(current_group[-1].transform.rotation.yaw) % 360.0
        if len(groups) > 0 and _ang_diff(first_yaw, last_yaw) < gap_deg:
            groups[0].extend(current_group)
        else:
            groups.append(current_group)

    return groups


def build_junction_groups(world_map, tl_landmarks):
    groups = defaultdict(list)
    no_junction = []
    for lm in tl_landmarks:
        jid = get_junction_id_for_landmark(world_map, lm)
        if jid is None:
            no_junction.append(lm)
        else:
            groups[jid].append(lm)
    return groups, no_junction


class TrafficLight_Controller:
    def __init__(self, phases, t_green=10.0, t_yellow=3.0, t_all_red=2.0, phase_offset=0.0):
        self.phases = phases
        self.tg = float(t_green)
        self.ty = float(t_yellow)
        self.tar = float(t_all_red)

        self.period = self.tg + self.ty + self.tar
        self.cycle = self.period * max(1, len(self.phases))
        self.time = float(phase_offset) % self.cycle

    def step(self, dt):
        self.time = (self.time + dt) % self.cycle

    def _active_phase(self):
        if not self.phases:
            return [], "R"

        idx = int(self.time // self.period) % len(self.phases)
        subt = self.time % self.period

        if subt < self.tg:
            return self.phases[idx], "G"
        if subt < self.tg + self.ty:
            return self.phases[idx], "Y"
        return [], "R"

    def state(self, tl_id):
        active_ids, color = self._active_phase()
        if int(tl_id) in active_ids:
            return color
        return "R"


class TrafficLight_Manager:
    def __init__(self, world_map, tl_landmarks, t_green=10.0, t_yellow=3.0, t_all_red=2.0,
                 ):
        self.controllers = {}
        self.id_to_junction = {}

        junction_groups, no_junction = build_junction_groups(world_map, tl_landmarks)

        for jid, lms in junction_groups.items():
            approach_groups = split_into_approaches(lms)
            phases = []
            for group in approach_groups:
                phase_ids = [int(lm.id) for lm in group]
                phases.append(phase_ids)

            num_phases = len(phases)
            period = t_green + t_yellow + t_all_red
            cycle_time = period * max(1, num_phases)
            random_offset = random.uniform(0, cycle_time)

            ctrl = TrafficLight_Controller(
                phases=phases,
                t_green=t_green,
                t_yellow=t_yellow,
                t_all_red=t_all_red,
                phase_offset=random_offset
            )

            self.controllers[int(jid)] = ctrl
            for lm in lms:
                self.id_to_junction[int(lm.id)] = int(jid)



    def step(self, dt):
        for ctrl in self.controllers.values():
            ctrl.step(dt)

        snapshot = self.capture_snapshot()

        return snapshot

    def state(self, tl_id):
        tl_id = int(tl_id)
        jid = self.id_to_junction.get(tl_id, None)
        if jid is None:
            return "R"
        ctrl = self.controllers.get(jid, None)
        return ctrl.state(tl_id) if ctrl is not None else "R"

    def capture_snapshot(self):
        snapshot = {}
        for tl_id in self.id_to_junction.keys():
            snapshot[int(tl_id)] = self.state(tl_id)
        return snapshot

    def get_snapshot(self, hist_idx=-1):
        if len(self.history) == 0:
            return {}

        idx = len(self.history) + hist_idx if hist_idx < 0 else hist_idx
        idx = max(0, min(idx, len(self.history) - 1))
        return self.history[idx]

    def get_snapshots(self, hist_indices):
        return [self.get_snapshot(i) for i in hist_indices]

def is_traffic_light_landmark(lm) -> bool:
    t = str(getattr(lm, "type", "")).strip().lower()
    name = str(getattr(lm, "name", "")).strip().lower().replace(" ", "")
    stype = str(getattr(lm, "sub_type", getattr(lm, "subtype", ""))).strip().lower().replace(" ", "")

    if t == "1000001":
        return True
    if "trafficlight" in name or "traffic_light" in name:
        return True
    if "trafficlight" in stype or "traffic_light" in stype:
        return True

    return False
def find_stop_wp_before_junction(wp, search_step=0.5, search_dist=60.0):
    if wp is None:
        return None

    cur = wp
    traveled = 0.0

    if cur.is_junction:
        while traveled < search_dist:
            prevs = cur.previous(search_step)
            if not prevs:
                return None
            prev_wp = prevs[0]
            if prev_wp is None:
                return None
            if not prev_wp.is_junction:
                return prev_wp
            cur = prev_wp
            traveled += search_step
        return None

    cur = wp
    traveled = 0.0
    while traveled < search_dist:
        nxts = cur.next(search_step)
        if not nxts:
            return None
        next_wp = nxts[0]
        if next_wp is None:
            return None

        if next_wp.is_junction:
            return cur

        cur = next_wp
        traveled += search_step

    return None




def tl_color_bgr(state):
    if state == "G":
        return (0, 255, 0)
    if state == "Y":
        return (0, 255, 255)
    return (0, 0, 255)


def get_stopline_vertices_from_landmark(world_map, lm, width_scale=0.5,
                                        search_step=0.5, search_dist=60.0):
    wp0 = getattr(lm, "waypoint", None)

    if wp0 is None:
        loc = lm.transform.location
        wp0 = world_map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp0 is None:
            return None, None, None

    stop_wp = find_stop_wp_before_junction(
        wp0,
        search_step=search_step,
        search_dist=search_dist,
    )
    if stop_wp is None:
        return None, None, None

    vec_forward = stop_wp.transform.get_forward_vector()
    vec_right = carla.Vector3D(x=-vec_forward.y, y=vec_forward.x, z=0.0)

    half_w = width_scale * stop_wp.lane_width
    center = stop_wp.transform.location

    loc_left = center - half_w * vec_right
    loc_right = center + half_w * vec_right
    return stop_wp, loc_left, loc_right

def make_stopline_key(stop_wp):
    loc = stop_wp.transform.location
    yaw = stop_wp.transform.rotation.yaw
    return (
        round(loc.x, 1),
        round(loc.y, 1),
        int(round(yaw / 10.0) * 10),
    )


# ── carla 객체 재구성 헬퍼 ─────────────────────────────────────────

def _make_location(d):
    return carla.Location(x=d["x"], y=d["y"], z=d["z"])


def _make_rotation(d):
    return carla.Rotation(pitch=d["pitch"], yaw=d["yaw"], roll=d["roll"])


def _make_transform(d):
    return carla.Transform(
        _make_location(d["location"]),
        _make_rotation(d["rotation"]),
    )


class _LandmarkProxy:

    def __init__(self, lm_dict):
        self.id = lm_dict["id"]
        self.transform = _make_transform(lm_dict["transform"])
        self.type = lm_dict.get("type", "")
        self.name = lm_dict.get("name", "")


class _StopWpProxy:

    def __init__(self, d):
        self.transform = _make_transform(d)
        self.lane_width = d.get("lane_width", 3.5)
        self.road_id = d.get("road_id", -1)
        self.lane_id = d.get("lane_id", 0)


class OfflineTLLoader:
    def __init__(self, data_dir, Town, t_green=10.0, t_yellow=3.0, t_all_red=2.0):
        json_path = os.path.join(data_dir, f"traffic_{Town}.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.town = data["town"]

        xodr_path = os.path.join(data_dir, f"{self.town}.xodr")
        with open(xodr_path, "r", encoding="utf-8") as f:
            xodr_content = f.read()
        self.world_map = carla.Map(self.town, xodr_content)

        self.tl_landmarks_proxy = []
        for lm_id_str, lm_info in data["landmarks"].items():
            self.tl_landmarks_proxy.append(_LandmarkProxy(lm_info))

        self.tl_stoplines = {}
        for lm_id_str, lm_info in data["landmarks"].items():
            sl = lm_info.get("stopline")
            if sl is None:
                continue
            tl_id = int(lm_id_str)
            self.tl_stoplines[tl_id] = {
                "stop_wp": _StopWpProxy(sl["stop_wp"]),
                "left":    _make_location(sl["left"]),
                "right":   _make_location(sl["right"]),
            }

        self.tl_manager = self._build_manager_from_json(
            data, t_green, t_yellow, t_all_red,
        )

    def _build_manager_from_json(self, data, t_green, t_yellow, t_all_red):
        mgr = TrafficLight_Manager.__new__(TrafficLight_Manager)
        mgr.controllers = {}
        mgr.id_to_junction = {}

        for jid_str, jinfo in data["junctions"].items():
            jid = int(jid_str)
            phases = jinfo["phases"]

            num_phases = len(phases)
            period = t_green + t_yellow + t_all_red
            cycle_time = period * max(1, num_phases)
            random_offset = random.uniform(0, cycle_time)

            ctrl = TrafficLight_Controller(
                phases=phases,
                t_green=t_green,
                t_yellow=t_yellow,
                t_all_red=t_all_red,
                phase_offset=random_offset,
            )
            mgr.controllers[jid] = ctrl

            for lm_id in jinfo["landmark_ids"]:
                mgr.id_to_junction[int(lm_id)] = jid

        return mgr
