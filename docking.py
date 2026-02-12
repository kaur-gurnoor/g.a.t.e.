# Artificial Intelligence tools were used to help with the tools

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, List
import math
import time

def norm3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)

def dot3(ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> float:
    return ax * bx + ay * by + az * bz

class DockState(str, Enum):
    FAR = "FAR"            
    APPROACH = "APPROACH"  
    HOLD = "HOLD"          
    CAPTURE = "CAPTURE"     
    DOCKED = "DOCKED"
    ABORT = "ABORT"     


@dataclass
class LVLHState:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float

    def distance(self) -> float:
        return norm3(self.x, self.y, self.z)

    def closing_speed(self) -> float:
        """
        Closing speed toward the station:
        v_close = -(r dot v)/||r||
        Positive means approaching, negative means receding.
        """
        d = self.distance()
        if d <= 1e-9:
            return float("inf")
        return -dot3(self.x, self.y, self.z, self.vx, self.vy, self.vz) / d


@dataclass
class Spacecraft:
    sc_id: str
    state: LVLHState
    arrival_eta_s: float       
    mission_priority: float       
    remaining_margin: float 
    present: bool = True
    dock_state: DockState = DockState.FAR
    last_update_ts: float = field(default_factory=lambda: time.time())

    # Runtime fields
    score: float = 0.0       


@dataclass
class DMSConfig:
    mu: float 
    a: float   

    min_separation_m: float = 30.0
    capture_zone_radius_m: float = 60.0        
    approach_zone_radius_m: float = 500.0     

    max_close_far_mps: float = 0.5          
    max_close_near_mps: float = 0.1    
    w_eta: float = 0.5
    w_priority: float = 2.0
    w_margin: float = 0.5
    w_queue: float = 0.2

    @property
    def mean_motion(self) -> float:
        return math.sqrt(self.mu / (self.a ** 3))


class DockingManagementSystem:
    def __init__(self, cfg: DMSConfig):
        self.cfg = cfg
        self.spacecraft: Dict[str, Spacecraft] = {}
        self.capture_zone_holder: Optional[str] = None

    def register(self, sc: Spacecraft) -> None:
        self.spacecraft[sc.sc_id] = sc

    def sensor_update(self, sc_id: str, new_state: LVLHState,
                      arrival_eta_s: Optional[float] = None,
                      remaining_margin: Optional[float] = None) -> None:
        """
        Replace this with real sensor fusion (lidar/camera/transponder).
        Here we just accept a state update.
        """
        sc = self.spacecraft[sc_id]
        sc.state = new_state
        sc.last_update_ts = time.time()
        if arrival_eta_s is not None:
            sc.arrival_eta_s = arrival_eta_s
        if remaining_margin is not None:
            sc.remaining_margin = remaining_margin

    def allowed_max_closing_speed(self, distance_m: float) -> float:
        """
        Simple distance-based closing speed limit (linear interpolation):
        - at approach boundary => max_close_far_mps
        - near capture zone => max_close_near_mps
        """
        cfg = self.cfg
        d0 = cfg.capture_zone_radius_m
        d1 = cfg.approach_zone_radius_m

        if distance_m <= d0:
            return cfg.max_close_near_mps
        if distance_m >= d1:
            return cfg.max_close_far_mps

        t = (distance_m - d0) / (d1 - d0)  # 0..1
        return cfg.max_close_near_mps + t * (cfg.max_close_far_mps - cfg.max_close_near_mps)

    def safety_check(self, sc: Spacecraft) -> Tuple[bool, str]:
        """
        Returns (safe, reason).
        """
        d = sc.state.distance()
        vclose = sc.state.closing_speed()

        if d < self.cfg.min_separation_m:
            return False, f"MIN_SEPARATION_VIOLATION d={d:.2f}m < {self.cfg.min_separation_m:.2f}m"

        vmax = self.allowed_max_closing_speed(d)
        if vclose > vmax:
            return False, f"CLOSING_SPEED_TOO_HIGH v_close={vclose:.3f}m/s > vmax={vmax:.3f}m/s at d={d:.1f}m"

        if d <= self.cfg.capture_zone_radius_m:
            if self.capture_zone_holder is None or self.capture_zone_holder == sc.sc_id:
                return True, "OK"
            return False, f"CAPTURE_ZONE_OCCUPIED by {self.capture_zone_holder}"

        return True, "OK"


    def compute_priority_scores(self) -> None:
        """
        Deterministic scoring function.
        Higher score => higher docking priority.
        """
        cfg = self.cfg
        queue_len = sum(1 for sc in self.spacecraft.values() if sc.present and sc.dock_state not in {DockState.DOCKED})

        for sc in self.spacecraft.values():
            if not sc.present or sc.dock_state == DockState.DOCKED:
                sc.score = -1e9
                continue

            eta_term = 1.0 / max(sc.arrival_eta_s, 1.0)

            sc.score = (
                cfg.w_eta * eta_term +
                cfg.w_priority * sc.mission_priority +
                cfg.w_margin * sc.remaining_margin -
                cfg.w_queue * queue_len
            )

    def choose_capture_candidate(self) -> Optional[Spacecraft]:
        """
        Select best candidate that is safe to enter capture zone.
        """
        self.compute_priority_scores()
        candidates: List[Spacecraft] = []

        for sc in self.spacecraft.values():
            if not sc.present or sc.dock_state in {DockState.DOCKED, DockState.ABORT}:
                continue

            d = sc.state.distance()
            if d > self.cfg.approach_zone_radius_m:
                continue

            safe, _ = self.safety_check(sc)
            candidates.append(sc)

        if not candidates:
            return None

        candidates.sort(key=lambda s: s.score, reverse=True)
        return candidates[0]

    def update_states(self) -> Dict[str, str]:
        """
        One control cycle: safety checks + scheduling + state transitions.
        Returns recommended actions (text) for each spacecraft.
        """
        actions: Dict[str, str] = {}

        for sc in self.spacecraft.values():
            if not sc.present or sc.dock_state == DockState.DOCKED:
                continue

            d = sc.state.distance()

            if d > self.cfg.approach_zone_radius_m:
                sc.dock_state = DockState.FAR
                actions[sc.sc_id] = "MONITOR_OUTSIDE_APPROACH_ZONE"
                continue

            safe, reason = self.safety_check(sc)
            if not safe:
                if "MIN_SEPARATION_VIOLATION" in reason:
                    sc.dock_state = DockState.ABORT
                    actions[sc.sc_id] = f"ABORT_RETREAT ({reason})"
                else:
                    sc.dock_state = DockState.HOLD
                    actions[sc.sc_id] = f"HOLD_POSITION ({reason})"
            else:
                if sc.dock_state in {DockState.FAR, DockState.HOLD}:
                    sc.dock_state = DockState.APPROACH
                    actions[sc.sc_id] = "CLEARED_TO_APPROACH"
                else:
                    actions.setdefault(sc.sc_id, "CONTINUE_CURRENT_MODE")

        candidate = self.choose_capture_candidate()

        if self.capture_zone_holder is None and candidate is not None:
            if candidate.state.distance() <= self.cfg.capture_zone_radius_m:
                self.capture_zone_holder = candidate.sc_id
                candidate.dock_state = DockState.CAPTURE
                actions[candidate.sc_id] = "ENTER_CAPTURE_ZONE_AND_BEGIN_EM_CAPTURE"

        if self.capture_zone_holder is not None:
            holder = self.spacecraft[self.capture_zone_holder]
            if holder.dock_state in {DockState.ABORT, DockState.DOCKED} or holder.state.distance() > self.cfg.capture_zone_radius_m:
                self.capture_zone_holder = None

        return actions

if __name__ == "__main__":
    cfg = DMSConfig(
        mu=6.26325e10,    
        a=5.0e5 + 4.73e5, 
        min_separation_m=30.0,
        capture_zone_radius_m=60.0,
        approach_zone_radius_m=500.0
    )

    dms = DockingManagementSystem(cfg)

    dms.register(Spacecraft(
        sc_id="CARGO-1",
        state=LVLHState(x=400, y=0, z=0, vx=-0.10, vy=0.0, vz=0.0),
        arrival_eta_s=1200,
        mission_priority=4.0,
        remaining_margin=8.0
    ))

    dms.register(Spacecraft(
        sc_id="CREW-7",
        state=LVLHState(x=450, y=20, z=0, vx=-0.12, vy=-0.01, vz=0.0),
        arrival_eta_s=900,
        mission_priority=9.0,
        remaining_margin=6.0
    ))

    print(f"Mean motion n = {cfg.mean_motion:.6e} rad/s (not required in simplified local control)")
    print("---- DMS cycles ----")

    
    for t in range(10):
        actions = dms.update_states()
        print(f"\nCycle {t}:")
        for sc_id, act in actions.items():
            sc = dms.spacecraft[sc_id]
            print(f"  {sc_id:8s} | d={sc.state.distance():6.1f}m | v_close={sc.state.closing_speed():.3f}m/s"
                  f" | state={sc.dock_state.value:8s} | score={sc.score:.3f} | action={act}")

        for sc in dms.spacecraft.values():
            if sc.dock_state in {DockState.APPROACH, DockState.CAPTURE}:
                sc.state.x += sc.state.vx * 5.0 
        time.sleep(0.1)