from typing import List, Optional, Tuple
from collections import deque
from sim.utility import world_to_pixel

import cv2
import math
import numpy as np

HISTORY_INDICES: Tuple[int, ...] = (-1, -5, -10, -15)
_DEQUE_MAXLEN: int = 16   # abs(min(HISTORY_INDICES)) + 여유
VEHICLE_LENGTH_M = 4.69
VEHICLE_WIDTH_M  = 2.0
HEIGHT_FILTER_M  = 4.0                     # ego 높이 기준 ± 필터 범위
HEIGHT_SENTINEL  = -9999.0                 # "차량 없음" 표시용 높이 값
 
# 레이어가 나뉘는 Town 목록
LAYERED_TOWNS = {"Town04", "Town05"}

def make_ego_local_mask(
    canvas_h: int,
    canvas_w: int,
    ppm: float,
    length_m: float = VEHICLE_LENGTH_M,
    width_m: float = VEHICLE_WIDTH_M,
    scale: float = 1.15,
    move_px: int = 0,
    yaw_rad: float = 0.0,
    center: Optional[Tuple[float, float]] = None,
    value: int = 255,
) -> np.ndarray:
    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
 
    length_px = max(1, int(np.round(length_m * scale * ppm)))
    width_px  = max(1, int(np.round(width_m  * scale * ppm)))
 
    if center is None:
        u0 = canvas_w / 2.0
        v0 = canvas_h / 2.0 + move_px
    else:
        u0, v0 = center
 
    half_l = length_px / 2.0
    half_w = width_px  / 2.0
 
    pts = np.array([
        [-half_w,  half_l],
        [ half_w,  half_l],
        [ half_w, -half_l],
        [-half_w, -half_l],
    ], dtype=np.float32)
 
    yaw_rad = yaw_rad + np.deg2rad(90)
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
 
    pts = pts @ R.T
    pts[:, 0] += u0
    pts[:, 1] += v0
    pts = np.round(pts).astype(np.int32)
    cv2.fillConvexPoly(mask, pts, value)
    return mask
 
 
def check_ego_collision(
    ego_local_mask: np.ndarray,
    ped_local_mask: np.ndarray,
    overlap_px_th: int = 1,
) -> Tuple[bool, Optional[str]]:
    ego_bin = ego_local_mask > 0
    ped_overlap_px = int(np.count_nonzero(ego_bin & (ped_local_mask > 0)))
    if ped_overlap_px >= overlap_px_th:
        return True
    return False
 
 
def check_ego_out_of_road(
    ego_local_mask: np.ndarray,
    das_local_bev: np.ndarray,
    out_of_road_ratio_th: float = 0.30,
) -> Tuple[bool, float]:
    ego_bin = ego_local_mask > 0
    ego_px  = int(np.count_nonzero(ego_bin))
    if ego_px == 0:
        return False, 0.0
    outside_px = int(np.count_nonzero(ego_bin & ~(das_local_bev > 0)))
    out_of_road_ratio = outside_px / float(ego_px)
    return out_of_road_ratio >= out_of_road_ratio_th, out_of_road_ratio
 
 
def draw_obb(mask, cx, cy, hl, hw, theta, value=255):
    c = math.cos(theta)
    s = math.sin(theta)
    pts = np.array([
        [ hl,  hw], [ hl, -hw],
        [-hl, -hw], [-hl,  hw],
    ], dtype=np.float32)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts = pts @ R.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    pts_i = np.round(pts).astype(np.int32)
    cv2.fillConvexPoly(mask, pts_i, int(value))
 
 
def ensure_u8_mask(mask):
    if mask.dtype == np.bool_:
        return mask.astype(np.uint8) * 255
    m = mask.astype(np.uint8)
    if m.max() <= 1:
        m = m * 255
    return m
 
 
def make_local_bev_ego_aligned(
    global_mask: np.ndarray,
    center_px: Tuple[float, float],
    ego_yaw_rad: float,
    out_size: int = 200,
    out_mpp: float = 0.20,
    pixels_per_meter: float = 5.0,
    fill: int = 0,
    move_px: int = 0,
) -> np.ndarray:
    u0 = out_size / 2.0
    v0 = out_size / 2.0 + move_px
    k  = out_mpp * pixels_per_meter
    c  = math.cos(ego_yaw_rad)
    s  = math.sin(ego_yaw_rad)
 
    M = np.array([
        [-s * k, -c * k, center_px[0] + k * (s * u0 + c * v0)],
        [ c * k, -s * k, center_px[1] + k * (-c * u0 + s * v0)],
    ], dtype=np.float32)
 
    # float32 height mask도 처리 가능하도록 flags 분기
    if global_mask.dtype == np.float32:
        flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
        border_val = float(fill)
    else:
        flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
        border_val = int(fill)
 
    return cv2.warpAffine(
        global_mask, M, (out_size, out_size),
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )
 
 
def _extract_ped_world_pos(ped) -> Tuple[Optional[float], Optional[float]]:
    """보행자 객체에서 월드 좌표 (x, y) 추출."""
    if hasattr(ped, "x") and hasattr(ped, "y"):
        return float(ped.x), float(ped.y)
    if isinstance(ped, dict):
        if "pos" in ped:
            return float(ped["pos"][0]), float(ped["pos"][1])
        if "x" in ped and "y" in ped:
            return float(ped["x"]), float(ped["y"])
    return None, None
 
 
def _extract_ped_z(ped) -> float:
    """보행자 객체에서 z 좌표 추출. 없으면 0.0."""
    if hasattr(ped, "z"):
        return float(ped.z)
    if isinstance(ped, dict) and "z" in ped:
        return float(ped["z"])
    if isinstance(ped, dict) and "pos" in ped and len(ped["pos"]) >= 3:
        return float(ped["pos"][2])
    return 0.0
 
 
# ═══════════════════════════════════════════════════════
#  VehicleFrame: 한 프레임의 글로벌 vehicle 맵
#  - instance_mask (uint16) : 픽셀별 vehicle id (0 = 비어 있음)
#  - height_mask   (float32): 픽셀별 vehicle z 값 (HEIGHT_SENTINEL = 비어 있음)
# ═══════════════════════════════════════════════════════
 
class VehicleFrame:
    __slots__ = ("instance_mask", "height_mask", "collided_vehicle_ids")

    def __init__(self, shape):
        self.instance_mask = np.zeros(shape, dtype=np.uint16)
        self.height_mask = np.full(shape, HEIGHT_SENTINEL, dtype=np.float32)
        self.collided_vehicle_ids = set()
    # ── 높이 값이 유효한 픽셀만 반환하는 편의 프로퍼티 ──
    @property
    def occupancy(self) -> np.ndarray:
        """bool mask: 차량이 존재하는 픽셀."""
        return self.instance_mask > 0
 
 
class PedFrame:
    """한 시점의 글로벌 ped 맵: occupancy + height."""
    __slots__ = ("occupancy_mask", "height_mask")

    def __init__(self, shape: Tuple[int, int]):
        self.occupancy_mask = np.zeros(shape, dtype=np.uint8)
        self.height_mask    = np.full(shape, HEIGHT_SENTINEL, dtype=np.float32)

def _obb_pts(cx, cy, hl, hw, theta):
    """OBB 꼭짓점 4개(int32) 반환."""
    c = math.cos(theta)
    s = math.sin(theta)
    pts = np.array([
        [ hl,  hw], [ hl, -hw],
        [-hl, -hw], [-hl,  hw],
    ], dtype=np.float32)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts = pts @ R.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return np.round(pts).astype(np.int32)


def _fill_height_obb(
    height_mask: np.ndarray,
    pts_i: np.ndarray,
    height_val: float,
) -> None:
    """
    height_mask(float32)에 OBB 영역을 height_val로 채운다.
    겹침 시 last-writer-wins.
    """
    h, w = height_mask.shape[:2]
    x_min = max(0, int(np.min(pts_i[:, 0])))
    x_max = min(w - 1, int(np.max(pts_i[:, 0])))
    y_min = max(0, int(np.min(pts_i[:, 1])))
    y_max = min(h - 1, int(np.max(pts_i[:, 1])))
    if x_min > x_max or y_min > y_max:
        return

    local_pts = pts_i.copy()
    local_pts[:, 0] -= x_min
    local_pts[:, 1] -= y_min

    roi_h = y_max - y_min + 1
    roi_w = x_max - x_min + 1
    obb_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillConvexPoly(obb_mask, local_pts, 255)

    height_mask[y_min:y_max + 1, x_min:x_max + 1][obb_mask > 0] = height_val



class GlobalDynamicMapManager:

    def __init__(
        self,
        global_map_shape: Tuple[int, int],
        town: str,
        maxlen: int = _DEQUE_MAXLEN,
    ):
        self.town = town
        self._shape: Tuple[int, int] = global_map_shape

        self._vehicle_history_high: deque[VehicleFrame] = deque(maxlen=maxlen)
        self._vehicle_history_low:  deque[VehicleFrame] = deque(maxlen=maxlen)
        self._ped_history: deque[PedFrame] = deque(maxlen=maxlen)

        self.ego_collision = False

    def push(
        self,
        vehicle_agents: list,
        ped_agents: list,
        offset_x: float, offset_y: float,
        ppm: float,
        veh_len_m: float,
        veh_wid_m: float,
        ped_box_px: int = 8,
    ) -> None:
        if self.town in LAYERED_TOWNS:
            vf_high = self._render_global_vehicle_frame(
                vehicle_agents, offset_x, offset_y, ppm,
                veh_len_m, veh_wid_m, layer_id=1,
            )
            self._vehicle_history_high.append(vf_high)

        vf_low = self._render_global_vehicle_frame(
            vehicle_agents, offset_x, offset_y, ppm,
            veh_len_m, veh_wid_m, layer_id=-1,
        )
        self._vehicle_history_low.append(vf_low)

        pf = self._render_global_ped_frame(
            ped_agents, offset_x, offset_y, ppm, ped_box_px,
        )
        self._ped_history.append(pf)

    def get_vehicle_bev_history(
        self,
        ego_pixel_x: float,
        ego_pixel_y: float,
        ego_z: float,
        ego_yaw_rad: float,
        ego_id: int,
        out_size: int,
        ppm: float,
        move_px: int = 0,
        ego_layer: int = -1,
        height_threshold: float = 4.0,
        include_opposite_layer: bool = False,
        return_collision: bool = False,   # 추가
        overlap_px_th: int = 1,           # 추가
        ego_mask_scale: float = 1.15,     # 추가
    ):
        if ego_layer >= 1 and self.town in LAYERED_TOWNS:
            primary_deque   = self._vehicle_history_high
            secondary_deque = self._vehicle_history_low
        else:
            primary_deque   = self._vehicle_history_low
            secondary_deque = (
                self._vehicle_history_high
                if self.town in LAYERED_TOWNS
                else None
            )

        zero = np.zeros((out_size, out_size), dtype=np.uint8)
        results: List[np.ndarray] = []
        collided_now = False

        ego_local_mask = None
        if return_collision:
            ego_local_mask = make_ego_local_mask(
                canvas_h=out_size,
                canvas_w=out_size,
                ppm=ppm,
                length_m=VEHICLE_LENGTH_M,
                width_m=VEHICLE_WIDTH_M,
                scale=ego_mask_scale,
                move_px=move_px,
                yaw_rad=0.0,
                value=255,
            )
            ego_bin = ego_local_mask > 0

        for hist_i, idx in enumerate(HISTORY_INDICES):
            abs_idx = abs(idx)

            if len(primary_deque) >= abs_idx:
                primary_bev = self._extract_filtered_veh_bev(
                    primary_deque[-abs_idx],
                    ego_pixel_x, ego_pixel_y, ego_yaw_rad,
                    ego_id, ego_z,
                    out_size, ppm, move_px,
                    apply_height_filter=False,
                    height_threshold=height_threshold,
                )
            else:
                primary_bev = zero

            if include_opposite_layer and secondary_deque is not None and len(secondary_deque) >= abs_idx:
                secondary_bev = self._extract_filtered_veh_bev(
                    secondary_deque[-abs_idx],
                    ego_pixel_x, ego_pixel_y, ego_yaw_rad,
                    ego_id, ego_z,
                    out_size, ppm, move_px,
                    apply_height_filter=True,
                    height_threshold=height_threshold,
                )
            else:
                secondary_bev = zero

            merged = np.where(
                (primary_bev > 0) | (secondary_bev > 0),
                np.uint8(255), np.uint8(0),
            )
            results.append(merged)

            # 현재 프레임(-1) 충돌 bool
            if return_collision and hist_i == 0:
                overlap_px = int(np.count_nonzero(ego_bin & (merged > 0)))
                collided_now = overlap_px >= overlap_px_th

        if return_collision:
            return results, collided_now
        return results

    def get_ped_bev_history(
        self,
        ego_pixel_x: float,
        ego_pixel_y: float,
        ego_z: float,
        ego_yaw_rad: float,
        out_size: int,
        ppm: float,
        move_px: int = 0,
        height_threshold: float = HEIGHT_FILTER_M,
    ) -> List[np.ndarray]:
        zero = np.zeros((out_size, out_size), dtype=np.uint8)
        results: List[np.ndarray] = []

        bev_params = dict(
            center_px=(ego_pixel_x, ego_pixel_y),
            ego_yaw_rad=ego_yaw_rad,
            out_size=out_size,
            out_mpp=1.0 / ppm,
            pixels_per_meter=ppm,
            move_px=move_px,
        )

        for idx in HISTORY_INDICES:
            abs_idx = abs(idx)
            if len(self._ped_history) < abs_idx:
                results.append(zero.copy())
                continue

            pf: PedFrame = self._ped_history[-abs_idx]

            local_occ = make_local_bev_ego_aligned(
                pf.occupancy_mask, fill=0, **bev_params,
            )
            occ = local_occ > 0

            if not np.any(occ):
                results.append(zero.copy())
                continue

            local_h = make_local_bev_ego_aligned(
                pf.height_mask, fill=HEIGHT_SENTINEL, **bev_params,
            )

            h_valid  = local_h > (HEIGHT_SENTINEL + 1.0)
            in_range = np.abs(local_h - ego_z) <= height_threshold
            filtered = occ & h_valid & in_range

            results.append(
                np.where(filtered, np.uint8(255), np.uint8(0))
            )

        return results

    def get_current_vehicle_global_mask(self, layer: int = -1) -> np.ndarray:
        deque_ = (self._vehicle_history_high if layer >= 1
                  else self._vehicle_history_low)
        if deque_:
            return deque_[-1].instance_mask
        return np.zeros(self._shape, dtype=np.uint16)

    def get_current_ped_global_mask(self) -> np.ndarray:
        if self._ped_history:
            return self._ped_history[-1].occupancy_mask
        return np.zeros(self._shape, dtype=np.uint8)

    def _extract_filtered_veh_bev(
        self,
        vf: VehicleFrame,
        ego_pixel_x: float,
        ego_pixel_y: float,
        ego_yaw_rad: float,
        ego_id: int,
        ego_z: float,
        out_size: int,
        ppm: float,
        move_px: int,
        apply_height_filter: bool,
        height_threshold: float,
    ) -> np.ndarray:
        bev_params = dict(
            center_px=(ego_pixel_x, ego_pixel_y),
            ego_yaw_rad=ego_yaw_rad,
            out_size=out_size,
            out_mpp=1.0 / ppm,
            pixels_per_meter=ppm,
            move_px=move_px,
        )

        local_inst = make_local_bev_ego_aligned(
            vf.instance_mask, fill=0, **bev_params,
        )
        local_inst[local_inst == ego_id] = 0
        occ = local_inst > 0

        if apply_height_filter and np.any(occ):
            local_h = make_local_bev_ego_aligned(
                vf.height_mask, fill=HEIGHT_SENTINEL, **bev_params,
            )
            h_valid  = local_h > (HEIGHT_SENTINEL + 1.0)
            in_range = np.abs(local_h - ego_z) <= height_threshold
            occ = occ & h_valid & in_range

        return np.where(occ, np.uint8(255), np.uint8(0))

    def _render_global_vehicle_frame(
        self,
        vehicle_agents: list,
        offset_x: float, offset_y: float,
        ppm: float,
        veh_len_m: float, veh_wid_m: float,
        layer_id: int,
    ) -> VehicleFrame:
        vf = VehicleFrame(self._shape)
        half_len_px = max(1, int(round(veh_len_m * ppm / 2.0)))
        half_wid_px = max(1, int(round(veh_wid_m * ppm / 2.0)))

        for agent in vehicle_agents:
            veh = agent["veh"] if isinstance(agent, dict) else agent
            alive = agent.get("alive", True) if isinstance(agent, dict) else True
            if not alive:
                continue

            agent_layer = (
                agent.get("layer", getattr(veh, "layer", -1))
                if isinstance(agent, dict)
                else getattr(veh, "layer", -1)
            )
            if layer_id != agent_layer:
                continue

            id_ = veh.id
            cx, cy = world_to_pixel(veh.x, veh.y, offset_x, offset_y, ppm)
            pts_i = _obb_pts(cx, cy, half_len_px, half_wid_px, veh.yaw)

            self._draw_obb_add(vf.instance_mask, pts_i, value=id_)
            _fill_height_obb(vf.height_mask, pts_i, float(veh.z))

        return vf

    @staticmethod
    def _draw_obb_add(mask, pts_i, value=1):
        h, w = mask.shape[:2]
        x_min = max(0, int(np.min(pts_i[:, 0])))
        x_max = min(w - 1, int(np.max(pts_i[:, 0])))
        y_min = max(0, int(np.min(pts_i[:, 1])))
        y_max = min(h - 1, int(np.max(pts_i[:, 1])))
        if x_min > x_max or y_min > y_max:
            return 0, []

        local_pts = pts_i.copy()
        local_pts[:, 0] -= x_min
        local_pts[:, 1] -= y_min

        roi_h = y_max - y_min + 1
        roi_w = x_max - x_min + 1
        tmp_roi = np.zeros((roi_h, roi_w), dtype=mask.dtype)
        cv2.fillConvexPoly(tmp_roi, local_pts, int(value))

        existing_roi = mask[y_min:y_max + 1, x_min:x_max + 1]
        overlap_mask = (tmp_roi > 0) & (existing_roi > 0)

        collided_ids = []
        if np.any(overlap_mask):
            collided_ids = list(np.unique(existing_roi[overlap_mask]).astype(int))

        mask[y_min:y_max + 1, x_min:x_max + 1] += tmp_roi
        pixel_count = int(np.count_nonzero(tmp_roi))
        return pixel_count, collided_ids

    def _render_global_ped_frame(
        self,
        ped_agents: list,
        offset_x: float, offset_y: float,
        ppm: float,
        box_px: int = 8,
    ) -> PedFrame:
        pf = PedFrame(self._shape)
        half = box_px // 2
        H, W = self._shape

        for ped in ped_agents:
            wx, wy = _extract_ped_world_pos(ped)
            if wx is None:
                continue

            pz = _extract_ped_z(ped)

            cx, cy = world_to_pixel(wx, wy, offset_x, offset_y, ppm)
            x0 = max(0, cx - half);  x1 = min(W, cx + half + 1)
            y0 = max(0, cy - half);  y1 = min(H, cy + half + 1)

            pf.occupancy_mask[y0:y1, x0:x1] = 255
            pf.height_mask[y0:y1, x0:x1]    = pz

        return pf