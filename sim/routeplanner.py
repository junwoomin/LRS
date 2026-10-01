import math
import carla
import numpy as np
import networkx as nx
from enum import Enum
from copy import deepcopy
from collections import deque
import random


class GlobalRoutePlannerDAO(object):
    def __init__(self, wmap, sampling_resolution):
        self._sampling_resolution = sampling_resolution
        self._wmap = wmap

    def get_topology(self):
        topology = []
        for segment in self._wmap.get_topology():
            wp1, wp2 = segment[0], segment[1]
            l1, l2   = wp1.transform.location, wp2.transform.location
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            wp1.transform.location, wp2.transform.location = l1, l2

            seg_dict = {}
            seg_dict['entry'], seg_dict['exit']       = wp1, wp2
            seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)
            seg_dict['path'] = []

            endloc = wp2.transform.location
            if wp1.transform.location.distance(endloc) > self._sampling_resolution:
                w = wp1.next(self._sampling_resolution)[0]
                while w.transform.location.distance(endloc) > self._sampling_resolution:
                    seg_dict['path'].append(w)
                    w = w.next(self._sampling_resolution)[0]
            else:
                seg_dict['path'].append(wp1.next(self._sampling_resolution / 2.0)[0])

            topology.append(seg_dict)
        return topology

    def get_waypoint(self, location):
        return self._wmap.get_waypoint(location)

    def get_resolution(self):
        return self._sampling_resolution


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance):
        self.saved_route           = deque()
        self.route                 = deque()
        self.saved_route_distances = deque()
        self.route_distances       = deque()
        self.min_distance          = min_distance
        self.max_distance          = max_distance
        self.is_last               = False
        self.mean                  = np.array([0.0, 0.0])
        self.scale                 = np.array([111324.60662786, 111319.490945])

    def set_route(self, global_plan, gps=False):
        self.route.clear()
        for pos, cmd in global_plan:
            if gps:
                pos  = np.array([pos['lat'], pos['lon']])
                pos -= self.mean
                pos *= self.scale
            else:
                pos  = np.array([pos.location.x, pos.location.y])
                pos -= self.mean
            self.route.append((pos, cmd))

        self.route_distances.append(0.0)
        for i in range(1, len(self.route)):
            diff     = self.route[i][0] - self.route[i - 1][0]
            distance = (diff[0] ** 2 + diff[1] ** 2) ** 0.5
            self.route_distances.append(distance)

    def run_step(self, gps):
        if len(self.route) <= 2:
            self.is_last = True
            return self.route

        to_pop            = 0
        farthest_in_range = -np.inf
        cumulative_dist   = 0.0

        for i in range(1, len(self.route)):
            if cumulative_dist > self.max_distance:
                break
            cumulative_dist += self.route_distances[i]
            diff     = self.route[i][0] - gps
            distance = (diff[0] ** 2 + diff[1] ** 2) ** 0.5
            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop            = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()
                self.route_distances.popleft()

        return self.route

    def save(self):
        self.saved_route           = deepcopy(self.route)
        self.saved_route_distances = deepcopy(self.route_distances)

    def load(self):
        self.route           = self.saved_route
        self.route_distances = self.saved_route_distances
        self.is_last         = False


class RoadOption(Enum):
    VOID           = -1
    LEFT           = 1
    RIGHT          = 2
    STRAIGHT       = 3
    LANEFOLLOW     = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT= 6


def vector(location_1, location_2):
    x    = location_2.x - location_1.x
    y    = location_2.y - location_1.y
    z    = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return [x / norm, y / norm, z / norm]


class GlobalRoutePlanner(object):
    def __init__(self, dao):
        self._dao                      = dao
        self._topology                 = None
        self._graph                    = None
        self._id_map                   = None
        self._road_id_to_edge          = None
        self._intersection_end_node    = -1
        self._previous_decision        = RoadOption.VOID

    def setup(self):
        self._topology = self._dao.get_topology()
        self._graph, self._id_map, self._road_id_to_edge = self._build_graph()
        self._find_loose_ends()
        self._lane_change_link()

    def _build_graph(self):
        graph          = nx.DiGraph()
        id_map         = {}
        road_id_to_edge = {}

        for segment in self._topology:
            entry_xyz, exit_xyz = segment['entryxyz'], segment['exitxyz']
            path                = segment['path']
            entry_wp, exit_wp   = segment['entry'], segment['exit']
            intersection        = entry_wp.is_intersection
            road_id, section_id, lane_id = entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id

            for vertex in entry_xyz, exit_xyz:
                if vertex not in id_map:
                    new_id          = len(id_map)
                    id_map[vertex]  = new_id
                    graph.add_node(new_id, vertex=vertex)

            n1 = id_map[entry_xyz]
            n2 = id_map[exit_xyz]
            road_id_to_edge.setdefault(road_id, {}).setdefault(section_id, {})[lane_id] = (n1, n2)

            entry_carla_vector = entry_wp.transform.rotation.get_forward_vector()
            exit_carla_vector  = exit_wp.transform.rotation.get_forward_vector()
            graph.add_edge(
                n1, n2,
                length=len(path) + 1, path=path,
                entry_waypoint=entry_wp, exit_waypoint=exit_wp,
                entry_vector=np.array([entry_carla_vector.x, entry_carla_vector.y, entry_carla_vector.z]),
                exit_vector =np.array([exit_carla_vector.x,  exit_carla_vector.y,  exit_carla_vector.z]),
                net_vector  =vector(entry_wp.transform.location, exit_wp.transform.location),
                intersection=intersection, type=RoadOption.LANEFOLLOW
            )

        return graph, id_map, road_id_to_edge

    def _find_loose_ends(self):
        count_loose_ends  = 0
        hop_resolution    = self._dao.get_resolution()

        for segment in self._topology:
            end_wp    = segment['exit']
            exit_xyz  = segment['exitxyz']
            road_id, section_id, lane_id = end_wp.road_id, end_wp.section_id, end_wp.lane_id

            if road_id in self._road_id_to_edge and \
               section_id in self._road_id_to_edge[road_id] and \
               lane_id in self._road_id_to_edge[road_id][section_id]:
                pass
            else:
                count_loose_ends += 1
                self._road_id_to_edge.setdefault(road_id, {}).setdefault(section_id, {})
                n1 = self._id_map[exit_xyz]
                n2 = -1 * count_loose_ends
                self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)

                next_wp = end_wp.next(hop_resolution)
                path    = []
                while next_wp and \
                      next_wp[0].road_id    == road_id and \
                      next_wp[0].section_id == section_id and \
                      next_wp[0].lane_id    == lane_id:
                    path.append(next_wp[0])
                    next_wp = next_wp[0].next(hop_resolution)

                if path:
                    n2_xyz = (path[-1].transform.location.x,
                              path[-1].transform.location.y,
                              path[-1].transform.location.z)
                    self._graph.add_node(n2, vertex=n2_xyz)
                    self._graph.add_edge(
                        n1, n2,
                        length=len(path) + 1, path=path,
                        entry_waypoint=end_wp, exit_waypoint=path[-1],
                        entry_vector=None, exit_vector=None, net_vector=None,
                        intersection=end_wp.is_intersection, type=RoadOption.LANEFOLLOW
                    )

    def _localize(self, location):
        waypoint = self._dao.get_waypoint(location)
        edge = None
        try:
            edge = self._road_id_to_edge[waypoint.road_id][waypoint.section_id][waypoint.lane_id]
        except KeyError:
            print("Failed to localize!", waypoint.road_id, waypoint.section_id,
                  waypoint.lane_id, waypoint.transform.location.x, waypoint.transform.location.y)
        return edge

    def _lane_change_link(self):
        for segment in self._topology:
            left_found = right_found = False
            for waypoint in segment['path']:
                if segment['entry'].is_intersection:
                    break

                if bool(waypoint.lane_change & carla.LaneChange.Right) and not right_found:
                    next_wp = waypoint.get_right_lane()
                    if next_wp and next_wp.lane_type == carla.LaneType.Driving and \
                       waypoint.road_id == next_wp.road_id:
                        next_seg = self._localize(next_wp.transform.location)
                        if next_seg:
                            self._graph.add_edge(
                                self._id_map[segment['entryxyz']], next_seg[0],
                                entry_waypoint=segment['entry'],
                                exit_waypoint=self._graph.edges[next_seg[0], next_seg[1]]['entry_waypoint'],
                                path=[], length=0, type=RoadOption.CHANGELANERIGHT,
                                change_waypoint=waypoint
                            )
                            right_found = True

                if bool(waypoint.lane_change & carla.LaneChange.Left) and not left_found:
                    next_wp = waypoint.get_left_lane()
                    if next_wp and next_wp.lane_type == carla.LaneType.Driving and \
                       waypoint.road_id == next_wp.road_id:
                        next_seg = self._localize(next_wp.transform.location)
                        if next_seg:
                            self._graph.add_edge(
                                self._id_map[segment['entryxyz']], next_seg[0],
                                entry_waypoint=segment['entry'],
                                exit_waypoint=self._graph.edges[next_seg[0], next_seg[1]]['entry_waypoint'],
                                path=[], length=0, type=RoadOption.CHANGELANELEFT,
                                change_waypoint=waypoint
                            )
                            left_found = True

                if left_found and right_found:
                    break

    def _distance_heuristic(self, n1, n2):
        l1 = np.array(self._graph.nodes[n1]['vertex'])
        l2 = np.array(self._graph.nodes[n2]['vertex'])
        return np.linalg.norm(l1 - l2)

    def _path_search(self, origin, destination):
        start, end = self._localize(origin), self._localize(destination)
        route = nx.astar_path(
            self._graph, source=start[0], target=end[0],
            heuristic=self._distance_heuristic, weight='length'
        )
        route.append(end[1])
        return route

    def _successive_last_intersection_edge(self, index, route):
        last_intersection_edge = None
        last_node = None
        for node1, node2 in [(route[i], route[i + 1]) for i in range(index, len(route) - 1)]:
            candidate_edge = self._graph.edges[node1, node2]
            if node1 == route[index]:
                last_intersection_edge = candidate_edge
            if candidate_edge['type'] == RoadOption.LANEFOLLOW and candidate_edge['intersection']:
                last_intersection_edge = candidate_edge
                last_node = node2
            else:
                break
        return last_node, last_intersection_edge

    def _turn_decision(self, index, route, threshold=math.radians(5)):
        decision      = None
        previous_node = route[index - 1]
        current_node  = route[index]
        next_node     = route[index + 1]
        next_edge     = self._graph.edges[current_node, next_node]

        if index > 0:
            if self._previous_decision != RoadOption.VOID and \
               self._intersection_end_node > 0 and \
               self._intersection_end_node != previous_node and \
               next_edge['type'] == RoadOption.LANEFOLLOW and \
               next_edge['intersection']:
                decision = self._previous_decision
            else:
                self._intersection_end_node = -1
                current_edge    = self._graph.edges[previous_node, current_node]
                calculate_turn  = (
                    current_edge['type'].value == RoadOption.LANEFOLLOW.value and
                    not current_edge['intersection'] and
                    next_edge['type'].value == RoadOption.LANEFOLLOW.value and
                    next_edge['intersection']
                )
                if calculate_turn:
                    last_node, tail_edge = self._successive_last_intersection_edge(index, route)
                    self._intersection_end_node = last_node
                    if tail_edge is not None:
                        next_edge = tail_edge

                    cv, nv = current_edge['exit_vector'], next_edge['net_vector']
                    cross_list = []
                    for neighbor in self._graph.successors(current_node):
                        sel_edge = self._graph.edges[current_node, neighbor]
                        if sel_edge['type'].value == RoadOption.LANEFOLLOW.value:
                            if neighbor != route[index + 1]:
                                sv = sel_edge['net_vector']
                                cross_list.append(np.cross(cv, sv)[2])

                    next_cross = np.cross(cv, nv)[2]
                    deviation  = math.acos(np.clip(
                        np.dot(cv, nv) / (np.linalg.norm(cv) * np.linalg.norm(nv)), -1.0, 1.0
                    ))

                    if not cross_list:
                        cross_list.append(0)

                    if deviation < threshold:
                        decision = RoadOption.STRAIGHT
                    elif cross_list and next_cross < min(cross_list):
                        decision = RoadOption.LEFT
                    elif cross_list and next_cross > max(cross_list):
                        decision = RoadOption.RIGHT
                    elif next_cross < 0:
                        decision = RoadOption.LEFT
                    elif next_cross > 0:
                        decision = RoadOption.RIGHT
                else:
                    decision = next_edge['type']
        else:
            decision = next_edge['type']

        self._previous_decision = decision
        return decision

    def abstract_route_plan(self, origin, destination):
        route = self._path_search(origin, destination)
        plan  = []
        for i in range(len(route) - 1):
            plan.append(self._turn_decision(i, route))
        return plan

    def _find_closest_in_list(self, current_waypoint, waypoint_list):
        min_dist      = float('inf')
        closest_index = -1
        for i, waypoint in enumerate(waypoint_list):
            distance = waypoint.transform.location.distance(current_waypoint.transform.location)
            if distance < min_dist:
                min_dist      = distance
                closest_index = i
        return closest_index

    def trace_route(self, origin, destination):
        route_trace         = []
        route               = self._path_search(origin, destination)
        current_waypoint    = self._dao.get_waypoint(origin)
        destination_waypoint = self._dao.get_waypoint(destination)
        resolution          = self._dao.get_resolution()

        for i in range(len(route) - 1):
            road_option = self._turn_decision(i, route)
            edge        = self._graph.edges[route[i], route[i + 1]]
            path        = []

            if edge['type'].value not in (RoadOption.LANEFOLLOW.value, RoadOption.VOID.value):
                route_trace.append((current_waypoint, road_option))
                exit_wp    = edge['exit_waypoint']
                n1, n2     = self._road_id_to_edge[exit_wp.road_id][exit_wp.section_id][exit_wp.lane_id]
                next_edge  = self._graph.edges[n1, n2]
                if next_edge['path']:
                    ci = self._find_closest_in_list(current_waypoint, next_edge['path'])
                    ci = min(len(next_edge['path']) - 1, ci + 5)
                    current_waypoint = next_edge['path'][ci]
                else:
                    current_waypoint = next_edge['exit_waypoint']
                route_trace.append((current_waypoint, road_option))
            else:
                path = [edge['entry_waypoint']] + edge['path'] + [edge['exit_waypoint']]
                ci   = self._find_closest_in_list(current_waypoint, path)
                for waypoint in path[ci:]:
                    current_waypoint = waypoint
                    route_trace.append((current_waypoint, road_option))
                    if len(route) - i <= 2 and \
                       waypoint.transform.location.distance(destination) < 2 * resolution:
                        break
                    elif len(route) - i <= 2 and \
                         current_waypoint.road_id    == destination_waypoint.road_id and \
                         current_waypoint.section_id == destination_waypoint.section_id and \
                         current_waypoint.lane_id    == destination_waypoint.lane_id:
                        di = self._find_closest_in_list(destination_waypoint, path)
                        if ci > di:
                            break

        return route_trace


# ---------------------------------------------------------------------------
# 경로 생성 유틸
# ---------------------------------------------------------------------------
def interpolate_trajectory(carla_map, trajectory_locations, max_len=2000, sampling_resolution=2.0):
    start_loc, end_loc = trajectory_locations[0], trajectory_locations[1]
    dao   = GlobalRoutePlannerDAO(carla_map, sampling_resolution)
    grp   = GlobalRoutePlanner(dao)
    grp.setup()
    route = grp.trace_route(start_loc, end_loc)
    raw_route = [(wp, ro) for wp, ro in route[:max_len]]
    return raw_route, raw_route


def extract_route_data(raw_route):
    route_data = []
    for wp, ro in raw_route:
        tf  = wp.transform
        x, y, z = tf.location.x, tf.location.y, tf.location.z
        yaw      = np.radians(tf.rotation.yaw)
        rid, sid, lid = int(wp.road_id), int(wp.section_id), int(wp.lane_id)

        landmarks = wp.get_landmarks(20.0, False)
        tl_id     = -1
        best_dist = float("inf")
        for lm in landmarks:
            if lm.type != '1000001':
                continue
            lm_tf = getattr(lm, "transform", None) or lm.get_transform()
            dist  = tf.location.distance(lm_tf.location)
            if dist < best_dist:
                best_dist = dist
                tl_id     = int(lm.id)

        junc_id = -1
        if wp.is_junction:
            junc = wp.get_junction()
            if junc is not None:
                junc_id = int(junc.id)

        route_data.append((
            float(x), float(y), float(z), float(yaw),
            rid, sid, lid, junc_id, tl_id, ro
        ))
    return route_data


def world_to_pix(loc_x, loc_y, offset_x, offset_y, W, H, ppm=5.0, invert_y=False):
    u = int(round((loc_x - offset_x) * ppm))
    v = int(round((loc_y - offset_y) * ppm))
    if invert_y:
        v = (H - 1) - v
    return u, v


def is_on_das(loc_x, loc_y, das_mask5, offset_x, offset_y, ppm=5.0, invert_y=False):
    H, W = das_mask5.shape[:2]
    u, v = world_to_pix(loc_x, loc_y, offset_x, offset_y, W=W, H=H, ppm=ppm, invert_y=invert_y)
    if u < 0 or u >= W or v < 0 or v >= H:
        return False
    return das_mask5[v, u] > 0


def generate_random_routes(world_map, das_mask5, offset_x, offset_y,
                            num_routes=1, sample_dist=2.0, max_len=4000,
                            max_tries=200, ppm=5.0, invert_y=False):
    all_waypoints         = world_map.generate_waypoints(sample_dist)
    non_junction_waypoints = [wp for wp in all_waypoints if not wp.is_junction]

    start_candidates = [
        wp for wp in non_junction_waypoints
        if is_on_das(wp.transform.location.x, wp.transform.location.y,
                     das_mask5, offset_x, offset_y, ppm=ppm, invert_y=invert_y)
    ]
    if len(start_candidates) < 10:
        start_candidates = non_junction_waypoints

    routes = []
    tries  = 0

    while len(routes) < num_routes and tries < max_tries:
        tries += 1
        if not start_candidates or len(non_junction_waypoints) < 2:
            break

        wp_start   = random.choice(start_candidates)
        wp_end     = random.choice(non_junction_waypoints)
        trajectory = [wp_start.transform.location, wp_end.transform.location]

        _, raw_route = interpolate_trajectory(world_map, trajectory, max_len=max_len,
                                              sampling_resolution=1.0)
        if not raw_route:
            continue
        del raw_route[1:5]
        processed_route = extract_route_data(raw_route)
        processed_route = smooth_route_lane_change(
            processed_route,
            jump_dist=3.0,
            curve_points=15,
            tangent_scale=0.35
        )

        if not processed_route:
            continue

        x0, y0 = processed_route[0][0], processed_route[0][1]
        if not is_on_das(x0, y0, das_mask5, offset_x, offset_y, ppm=ppm, invert_y=invert_y):
            continue

        routes.append(processed_route)

    return routes
def _norm(vx, vy, eps=1e-8):
    n = math.hypot(vx, vy)
    if n < eps:
        return 0.0, 0.0
    return vx / n, vy / n

def _estimate_heading(route, idx):
    """
    route[idx]에서의 진행 방향 추정
    route 원소는 [x, y, ...] 또는 tuple/list라고 가정
    """
    n = len(route)

    if n < 2:
        return 1.0, 0.0

    if idx <= 0:
        x0, y0 = route[0][0], route[0][1]
        x1, y1 = route[1][0], route[1][1]
        return _norm(x1 - x0, y1 - y0)

    if idx >= n - 1:
        x0, y0 = route[n - 2][0], route[n - 2][1]
        x1, y1 = route[n - 1][0], route[n - 1][1]
        return _norm(x1 - x0, y1 - y0)

    x_prev, y_prev = route[idx - 1][0], route[idx - 1][1]
    x_next, y_next = route[idx + 1][0], route[idx + 1][1]
    return _norm(x_next - x_prev, y_next - y_prev)

def hermite_curve(p0, p1, t0, t1, n_points=20, tangent_scale=3.0):
    """
    p0, p1: (x, y)
    t0, t1: 단위 tangent 벡터
    tangent_scale: tangent 길이 스케일
    """
    x0, y0 = p0
    x1, y1 = p1

    dist = math.hypot(x1 - x0, y1 - y0)
    m0 = (t0[0] * dist * tangent_scale, t0[1] * dist * tangent_scale)
    m1 = (t1[0] * dist * tangent_scale, t1[1] * dist * tangent_scale)

    out = []
    for i in range(n_points):
        s = i / (n_points - 1)

        h00 =  2*s**3 - 3*s**2 + 1
        h10 =      s**3 - 2*s**2 + s
        h01 = -2*s**3 + 3*s**2
        h11 =      s**3 -   s**2

        x = h00*x0 + h10*m0[0] + h01*x1 + h11*m1[0]
        y = h00*y0 + h10*m0[1] + h01*y1 + h11*m1[1]
        out.append([x, y])

    return out

def smooth_route_lane_change(route, jump_dist=3.0, curve_points=20, tangent_scale=0.35):
    """
    route: [[x,y,...], [x,y,...], ...]
    jump_dist보다 큰 점프가 나오면 그 구간을 spline으로 대체
    """
    if len(route) < 3:
        return route

    smoothed = [list(route[0])]

    for i in range(1, len(route)):
        prev_pt = route[i - 1]
        curr_pt = route[i]

        x0, y0 = prev_pt[0], prev_pt[1]
        x1, y1 = curr_pt[0], curr_pt[1]

        d = math.hypot(x1 - x0, y1 - y0)

        # 정상 구간이면 그대로 추가
        if d < jump_dist:
            smoothed.append(list(curr_pt))
            continue

        # 점프 구간이면 spline 삽입
        t0 = _estimate_heading(route, i - 1)
        t1 = _estimate_heading(route, i)

        curve_xy = hermite_curve(
            (x0, y0), (x1, y1),
            t0, t1,
            n_points=curve_points,
            tangent_scale=tangent_scale
        )

        # 첫 점은 prev와 중복되니 제외
        for k in range(1, len(curve_xy)):
            x, y = curve_xy[k]

            # route 포맷 유지
            new_pt = list(curr_pt)
            new_pt[0] = x
            new_pt[1] = y
            smoothed.append(new_pt)

    return smoothed


def generate_random_routes_roop(world_map, start_point, num_routes=1,
                                 sample_dist=2.0, max_len=4000, max_tries=100):
    """
    start_point: carla.Location  ← 현재 NPC 위치를 넘겨야 함.
    """
    all_waypoints = world_map.generate_waypoints(sample_dist)
    routes  = []
    tries   = 0

    while len(routes) < num_routes and tries < max_tries:
        tries += 1
        wps        = random.sample(all_waypoints, 1)
        trajectory = [start_point, wps[0].transform.location]
        _, raw_route = interpolate_trajectory(world_map, trajectory, max_len=max_len,
                                              sampling_resolution=1.0)
        if raw_route:
            routes.append(extract_route_data(raw_route))

    return routes
