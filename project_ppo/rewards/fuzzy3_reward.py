#!/usr/bin/env python3
"""
3-Input Sugeno Fuzzy Reward for TurtleBot3 Small House Navigation

INPUTS (3):
  A) Progress per step: Δd = d_prev - d_cur (meters/step)
  B) Heading error: e = abs(wrap_to_pi(theta_goal - yaw_robot)) (radians)
  C) Front obstacle clearance: c_front = min(LiDAR front sector) (meters)

OUTPUTS (Sugeno constants):
  VeryBad = -1.0
  Bad     = -0.5
  Neutral = -0.05  (mild time pressure)
  Good    = +0.3
  VeryGood= +0.6

TERMINALS:
  arrive=True  -> +120
  done=True    -> -100

PHYSICS-BASED CONSTRAINTS:
  v_max = 0.25 m/s, omega_max = 1.0 rad/s
  steps/sec ≈ 5.0 (measured)
  dt ≈ 0.2 s -> max_translation ≈ 0.05 m/step
"""

import math
from collections import deque
from typing import Sequence, Optional


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi


class Fuzzy3Reward:
    """
    3-input Sugeno fuzzy reward for navigation.
    """

    def __init__(self):
        # Physics constraints
        self.v_max = 0.25          # m/s
        self.omega_max = 1.0       # rad/s
        self.steps_per_sec = 5.0   # measured
        self.dt = 1.0 / self.steps_per_sec  # ~0.2 s
        self.max_translation = self.v_max * self.dt  # ~0.05 m/step

        # Membership parameters
        self.eps = 0.005   # progress deadzone (m)
        self.c_max = 2.0   # max clearance considered (m)

        # Sugeno output levels
        self.outputs = {
            "VeryBad": -1.0,
            "Bad": -0.5,
            "Neutral": -0.05,
            "Good": 0.3,
            "VeryGood": 0.6,
        }

        # Previous state for progress
        self.prev_distance: Optional[float] = None

        # Lightweight stats (optional)
        self.step_count = 0
        self.stats = {
            "progress": deque(maxlen=1000),
            "heading_error": deque(maxlen=1000),
            "front_clearance": deque(maxlen=1000),
            "nav_reward": deque(maxlen=1000),
        }

    def reset(self, initial_distance: float):
        self.prev_distance = float(initial_distance)

    # ----------------- Membership primitives -----------------
    def _trapezoid(self, x: float, a: float, b: float, c: float, d: float) -> float:
        """
        Trapezoidal membership function with shoulder handling.

        - left shoulder:  a==b
        - right shoulder: c==d
        """
        # Outside support
        if x < a or x > d:
            return 0.0

        # Left shoulder (a==b): immediate 1.0 from [a..c]
        if a == b:
            if a <= x <= c:
                return 1.0
            if c < x < d:
                return (d - x) / (d - c) if d != c else 0.0
            return 0.0

        # Right shoulder (c==d): 1.0 from [b..d]
        if c == d:
            if a <= x < b:
                return (x - a) / (b - a) if b != a else 0.0
            if b <= x <= d:
                return 1.0
            return 0.0

        # Normal trapezoid
        if a <= x < b:
            return (x - a) / (b - a) if b != a else 0.0
        if b <= x <= c:
            return 1.0
        if c < x <= d:
            return (d - x) / (d - c) if d != c else 0.0
        return 0.0

    def _triangle(self, x: float, a: float, b: float, c: float) -> float:
        """Triangular membership function."""
        if x <= a or x >= c:
            return 0.0
        if a < x < b:
            return (x - a) / (b - a) if b != a else 0.0
        # b <= x < c
        return (c - x) / (c - b) if c != b else 0.0

    # ----------------- INPUT A: Progress (Δd) -----------------
    def _progress_away(self, delta_d: float) -> float:
        """Away: trapezoid [-dmax, -dmax, -eps, 0]"""
        return self._trapezoid(delta_d, -self.max_translation, -self.max_translation, -self.eps, 0.0)

    def _progress_zero(self, delta_d: float) -> float:
        """Zero: triangle [-eps, 0, +eps]"""
        return self._triangle(delta_d, -self.eps, 0.0, self.eps)

    def _progress_toward(self, delta_d: float) -> float:
        """Toward: trapezoid [0, +eps, +dmax, +dmax]"""
        return self._trapezoid(delta_d, 0.0, self.eps, self.max_translation, self.max_translation)

    # ----------------- INPUT B: Heading error (e) -----------------
    def _heading_aligned(self, e: float) -> float:
        """Aligned: trapezoid [0, 0, 0.20, 0.35] rad (~0–20°)"""
        return self._trapezoid(e, 0.0, 0.0, 0.20, 0.35)

    def _heading_medium(self, e: float) -> float:
        """Medium: triangle [0.25, 0.80, 1.40] rad"""
        return self._triangle(e, 0.25, 0.80, 1.40)

    def _heading_misaligned(self, e: float) -> float:
        """Misaligned: trapezoid [1.00, 1.40, pi, pi] rad"""
        return self._trapezoid(e, 1.00, 1.40, math.pi, math.pi)

    # ----------------- INPUT C: Front clearance (c_front) -----------------
    # Your requirement:
    #   "I want to have until 0.6 danger"
    # Implemented as:
    #   Danger: trapezoid [0.00, 0.00, 0.60, 0.80]  -> 1.0 until 0.60, then fades to 0 by 0.80
    # And aligned overlap sets:
    #   Caution: triangle [0.60, 0.90, 1.20]
    #   Clear:   trapezoid [1.05, 1.30, c_max, c_max]
    def _clearance_danger(self, c: float) -> float:
        """Danger: trapezoid [0.00, 0.00, 0.60, 0.80] m (plateau until 0.60m)."""
        return self._trapezoid(c, 0.00, 0.00, 0.60, 0.80)

    def _clearance_caution(self, c: float) -> float:
        """Caution: triangle [0.60, 0.90, 1.20] m."""
        return self._triangle(c, 0.60, 0.90, 1.20)

    def _clearance_clear(self, c: float) -> float:
        """Clear: trapezoid [1.05, 1.30, c_max, c_max] m."""
        return self._trapezoid(c, 1.05, 1.30, self.c_max, self.c_max)

    def _compute_front_clearance(self, scan_range: Optional[Sequence[float]]) -> float:
        """
        TurtleBot3 Burger LiDAR: 10 samples spanning [-90°, +90°].
        Front sector indices [3,4,5,6,7] ≈ [-36°, +36°].
        """
        if scan_range is None:
            return 0.2
        try:
            N = len(scan_range)
        except Exception:
            return 0.2
        if N == 0:
            return 0.2

        if N == 10:
            front_indices = [3, 4, 5, 6, 7]
        else:
            center = N // 2
            k = max(1, N // 10)  # ~10% on each side
            front_indices = list(range(max(0, center - k), min(N, center + k + 1)))

        front_readings = []
        for i in front_indices:
            if 0 <= i < N:
                try:
                    front_readings.append(float(scan_range[i]))
                except Exception:
                    pass

        c_front = min(front_readings) if front_readings else 0.2
        return min(c_front, self.c_max)

    # ----------------- Rules + Sugeno -----------------
    def _evaluate_rules(self, delta_d: float, e: float, c_front: float):
        """
        Evaluate 10 fuzzy rules. Returns list of (firing_strength, output_value).
        """
        # Memberships
        prog_away = self._progress_away(delta_d)
        prog_zero = self._progress_zero(delta_d)
        prog_toward = self._progress_toward(delta_d)

        head_aligned = self._heading_aligned(e)
        head_medium = self._heading_medium(e)
        head_misaligned = self._heading_misaligned(e)

        clear_danger = self._clearance_danger(c_front)
        clear_caution = self._clearance_caution(c_front)
        clear_clear = self._clearance_clear(c_front)

        rules = []

        # R1: ANY, ANY, Danger -> VeryBad (safety override)
        rules.append((clear_danger, self.outputs["VeryBad"]))

        # R2: Away, ANY, Caution -> Bad
        rules.append((min(prog_away, clear_caution), self.outputs["Bad"]))

        # R3: ANY, Misaligned, Caution -> Bad
        rules.append((min(head_misaligned, clear_caution), self.outputs["Bad"]))

        # R4: Toward, Aligned, Clear -> VeryGood
        rules.append((min(prog_toward, head_aligned, clear_clear), self.outputs["VeryGood"]))

        # R5: Toward, Medium, Clear -> Good
        rules.append((min(prog_toward, head_medium, clear_clear), self.outputs["Good"]))

        # R6: Toward, Aligned, Caution -> Good
        rules.append((min(prog_toward, head_aligned, clear_caution), self.outputs["Good"]))

        # R7: Toward, Misaligned, Clear -> Neutral
        rules.append((min(prog_toward, head_misaligned, clear_clear), self.outputs["Neutral"]))

        # R8: Zero, Aligned, Clear -> Neutral
        rules.append((min(prog_zero, head_aligned, clear_clear), self.outputs["Neutral"]))

        # R9: Zero, Misaligned, ANY -> Bad
        rules.append((min(prog_zero, head_misaligned), self.outputs["Bad"]))

        # R10: Away, ANY, Clear -> Bad
        rules.append((min(prog_away, clear_clear), self.outputs["Bad"]))

        return rules

    def _defuzzify_sugeno(self, rules) -> float:
        """Sugeno weighted average of constants."""
        num = 0.0
        den = 0.0
        for w, z in rules:
            num += w * z
            den += w
        return self.outputs["Neutral"] if den < 1e-9 else (num / den)

    # ----------------- Public API -----------------
    def compute(
        self,
        current_distance: float,
        current_yaw: float,         # degrees
        current_rel_theta: float,   # degrees
        scan_range: Optional[Sequence[float]],
        done: bool,
        arrive: bool,
    ) -> float:
        if done:
            return -100.0
        if arrive:
            return 120.0

        # Safety: if reset() was not called
        if self.prev_distance is None:
            self.prev_distance = float(current_distance)

        # degrees -> radians
        yaw_rad = math.radians(float(current_yaw))
        theta_rad = math.radians(float(current_rel_theta))

        # Inputs
        delta_d = float(self.prev_distance) - float(current_distance)
        e = abs(wrap_to_pi(theta_rad - yaw_rad))
        c_front = self._compute_front_clearance(scan_range)

        # Inference
        rules = self._evaluate_rules(delta_d, e, c_front)
        nav_reward = self._defuzzify_sugeno(rules)

        # Stats
        self.step_count += 1
        self.stats["progress"].append(delta_d)
        self.stats["heading_error"].append(e)
        self.stats["front_clearance"].append(c_front)
        self.stats["nav_reward"].append(nav_reward)

        # Update
        self.prev_distance = float(current_distance)
        return nav_reward
