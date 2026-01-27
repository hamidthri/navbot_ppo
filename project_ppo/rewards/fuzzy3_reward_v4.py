#!/usr/bin/env python3
"""
Fuzzy3RewardV4: Enhanced 3-Input Sugeno Fuzzy Reward with Anti-Stall (Soft)

Fixes vs V3:
1) Normalized progress: p = clip(delta_d / d_scale, [-1, 1]) with d_scale=0.04
2) Anti-stall: penalize stall (Δd≈0) when safe and already aligned/medium
3) Conditional danger: penalize forward motion in danger strongly; allow rotate-in-place in danger if misaligned
4) Robust front clearance: use 20th percentile of front sector with filtering (finite, >0)
5) Earlier danger for clutter: Danger plateau until 0.35m, fades out by 0.55m
6) Configurable theta handling: theta_is_relative flag

Inputs:
A) Progress p (normalized from Δd)
B) Heading error e
C) Front clearance c_front

Sugeno outputs:
VeryBad  = -1.00
Bad      = -0.60
StallBad = -0.25
Neutral  = -0.10
Good     = +0.35
VeryGood = +0.85

Terminals:
arrive   -> +120
done     -> collision_penalty if provided else -100 (keeps env default)
timeout  -> handled in PPO (recommended -80)
"""

import math
from collections import deque
from typing import Optional, Sequence


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi


class Fuzzy3RewardV4:
    """
    Enhanced 3-input Sugeno fuzzy reward with anti-stall and robust clearance.
    """

    def __init__(self, theta_is_relative: bool = True, collision_penalty: Optional[float] = None):
        """
        Args:
            theta_is_relative:
                If True, current_rel_theta is already the relative angle-to-goal (recommended).
                If False, current_rel_theta is absolute goal angle and yaw must be subtracted.
            collision_penalty:
                If not None, overrides done() terminal reward. If None, returns -100 (env default).
        """
        # Physics (kept for context / reference)
        self.v_max = 0.25
        self.steps_per_sec = 5.0
        self.dt = 1.0 / self.steps_per_sec
        self.max_translation = self.v_max * self.dt  # ~0.05 m/step

        # Normalization for progress: Δd / d_scale -> p
        self.d_scale = 0.04  # m/step

        # Clearance clamp
        self.c_max = 2.0

        # Sugeno constants
        self.outputs = {
            "VeryBad": -1.00,
            "Bad": -0.60,
            "StallBad": -0.25,
            "Neutral": -0.10,
            "Good": 0.35,
            "VeryGood": 0.85,
        }

        # Terminal rewards
        self.arrive_reward = 120.0
        self.collision_penalty = collision_penalty  # None -> keep env default (-100)

        # Theta handling
        self.theta_is_relative = theta_is_relative
        print(f"[Fuzzy3RewardV4] theta_is_relative={theta_is_relative}", flush=True)

        # Previous distance for Δd
        self.prev_distance: Optional[float] = None

        # Optional stats buffer
        self.step_count = 0
        self.stats = {
            "progress_norm": deque(maxlen=2000),
            "heading_error": deque(maxlen=2000),
            "front_clearance": deque(maxlen=2000),
            "nav_reward": deque(maxlen=2000),
        }

    def reset(self, initial_distance: float):
        self.prev_distance = float(initial_distance)

    # ----------------- Membership primitives -----------------
    def _trapezoid(self, x: float, a: float, b: float, c: float, d: float) -> float:
        if x < a or x > d:
            return 0.0

        # Left shoulder
        if a == b:
            if a <= x <= c:
                return 1.0
            if c < x < d:
                return (d - x) / (d - c) if d != c else 0.0
            return 0.0

        # Right shoulder
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
        if x <= a or x >= c:
            return 0.0
        if a < x < b:
            return (x - a) / (b - a) if b != a else 0.0
        return (c - x) / (c - b) if c != b else 0.0

    # ----------------- INPUT A: Progress memberships on p -----------------
    def _p_away(self, p: float) -> float:
        """Away: trapezoid [-1, -1, -0.15, 0]"""
        return self._trapezoid(p, -1.0, -1.0, -0.15, 0.0)

    def _p_stall(self, p: float) -> float:
        """Stall: triangle [-0.10, 0, 0.10]"""
        return self._triangle(p, -0.10, 0.0, 0.10)

    def _p_toward_small(self, p: float) -> float:
        """TowardSmall: triangle [0.00, 0.25, 0.55]"""
        return self._triangle(p, 0.00, 0.25, 0.55)

    def _p_toward_good(self, p: float) -> float:
        """TowardGood: trapezoid [0.45, 0.70, 1.0, 1.0]"""
        return self._trapezoid(p, 0.45, 0.70, 1.0, 1.0)

    # ----------------- INPUT B: Heading error memberships -----------------
    # (Using the V4 numbers that match the earlier analysis)
    def _h_aligned(self, e: float) -> float:
        """Aligned: trapezoid [0, 0, 0.22, 0.40] rad"""
        return self._trapezoid(e, 0.0, 0.0, 0.22, 0.40)

    def _h_medium(self, e: float) -> float:
        """Medium: triangle [0.30, 0.90, 1.55] rad"""
        return self._triangle(e, 0.30, 0.90, 1.55)

    def _h_misaligned(self, e: float) -> float:
        """Misaligned: trapezoid [1.15, 1.65, pi, pi] rad"""
        return self._trapezoid(e, 1.15, 1.65, math.pi, math.pi)

    # ----------------- INPUT C: Clearance memberships -----------------
    def _c_danger(self, c: float) -> float:
        """Danger: trapezoid [0.00, 0.00, 0.35, 0.55]"""
        return self._trapezoid(c, 0.00, 0.00, 0.35, 0.55)

    def _c_caution(self, c: float) -> float:
        """Caution: triangle [0.45, 0.90, 1.30]"""
        return self._triangle(c, 0.45, 0.90, 1.30)

    def _c_clear(self, c: float) -> float:
        """Clear: trapezoid [1.10, 1.45, c_max, c_max]"""
        return self._trapezoid(c, 1.10, 1.45, self.c_max, self.c_max)

    def _compute_front_clearance(self, scan_range: Optional[Sequence[float]]) -> float:
        """
        Robust front clearance:
        - Take front sector
        - Filter invalid readings (nan/inf/<=0)
        - Use 20th percentile by sorted index
        """
        if not scan_range:
            return 0.2

        N = len(scan_range)
        if N == 10:
            front_indices = [3, 4, 5, 6, 7]
        else:
            center = N // 2
            k = max(1, N // 10)
            front_indices = range(max(0, center - k), min(N, center + k + 1))

        vals = []
        for i in front_indices:
            try:
                x = float(scan_range[i])
                if math.isfinite(x) and x > 0.0:
                    vals.append(x)
            except Exception:
                pass

        if not vals:
            return 0.2

        vals.sort()
        q = int(0.20 * (len(vals) - 1))
        c_front = vals[q]
        return max(0.0, min(c_front, self.c_max))

    # ----------------- Rules + Sugeno -----------------
    def _evaluate_rules(self, p: float, e: float, c_front: float):
        """
        Returns list of (firing_strength, output_value).
        """
        # Progress
        PA = self._p_away(p)
        PS = self._p_stall(p)
        P_small = self._p_toward_small(p)
        P_good = self._p_toward_good(p)
        PF = max(P_small, P_good)   # any forward progress

        # Heading
        HA = self._h_aligned(e)
        HM = self._h_medium(e)
        Hm = self._h_misaligned(e)

        # Clearance
        CD = self._c_danger(c_front)
        CC = self._c_caution(c_front)
        CL = self._c_clear(c_front)

        rules = []

        # Always-on baseline (time pressure + coverage)
        rules.append((0.02, self.outputs["Neutral"]))

        # ---------------- Danger: conditional ----------------
        # Forward motion in danger is very bad
        rules.append((min(CD, PF), self.outputs["VeryBad"]))
        # Rotate-in-place (stall) is OK if misaligned (needs to turn to escape)
        rules.append((min(CD, Hm, PS), self.outputs["Neutral"]))
        # But stall in danger while aligned/medium means "stuck" -> penalize
        rules.append((min(CD, max(HA, HM), PS), self.outputs["StallBad"]))
        # If moving away from goal in danger, don't crush it (escape behavior)
        rules.append((min(CD, PA), self.outputs["Neutral"]))

        # ---------------- Clear: anti-stall + progress ----------------
        # Anti-stall when safe and already oriented
        rules.append((min(CL, max(HA, HM), PS), self.outputs["StallBad"]))
        # Allow turning-in-place when misaligned (no penalty)
        rules.append((min(CL, Hm, PS), self.outputs["Neutral"]))

        # Reward progress strongly when aligned in clear space
        rules.append((min(CL, HA, P_good), self.outputs["VeryGood"]))
        rules.append((min(CL, HA, PF), self.outputs["Good"]))
        rules.append((min(CL, HM, P_good), self.outputs["Good"]))
        rules.append((min(CL, HM, PF), self.outputs["Neutral"]))
        # If misaligned but making progress, keep neutral (don’t reward blindly)
        rules.append((min(CL, Hm, PF), self.outputs["Neutral"]))

        # ---------------- Caution: smaller rewards + anti-stall ----------------
        rules.append((min(CC, HA, P_good), self.outputs["Good"]))
        rules.append((min(CC, max(HA, HM), PF), self.outputs["Neutral"]))
        rules.append((min(CC, max(HA, HM), PS), self.outputs["StallBad"]))

        # ---------------- Away (not in danger) ----------------
        rules.append((min(max(CL, CC), PA), self.outputs["Bad"]))

        return rules

    def _defuzzify_sugeno(self, rules) -> float:
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
        # Terminal rewards
        if arrive:
            return self.arrive_reward
        if done:
            return float(self.collision_penalty) if self.collision_penalty is not None else -100.0

        # Safety if reset() not called
        if self.prev_distance is None:
            self.prev_distance = float(current_distance)

        # Convert degrees -> radians
        yaw_rad = math.radians(float(current_yaw))
        theta_rad = math.radians(float(current_rel_theta))

        # Progress
        delta_d = float(self.prev_distance) - float(current_distance)
        p = max(-1.0, min(1.0, delta_d / self.d_scale))

        # Heading error
        if self.theta_is_relative:
            e = abs(wrap_to_pi(theta_rad))
        else:
            e = abs(wrap_to_pi(theta_rad - yaw_rad))

        # Robust clearance
        c_front = self._compute_front_clearance(scan_range)

        # Inference
        rules = self._evaluate_rules(p, e, c_front)
        nav_reward = self._defuzzify_sugeno(rules)

        # Stats
        self.step_count += 1
        self.stats["progress_norm"].append(p)
        self.stats["heading_error"].append(e)
        self.stats["front_clearance"].append(c_front)
        self.stats["nav_reward"].append(nav_reward)

        # Update
        self.prev_distance = float(current_distance)
        return nav_reward
