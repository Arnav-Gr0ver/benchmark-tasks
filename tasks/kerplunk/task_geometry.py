"""The demo's grip on the blue rod, in numbers check_physics.py can use without yam_arena (task.py imports these)."""

from __future__ import annotations

import math

import tower

LIFT_BIAS = 0.002                                                        # the held end lifted 2 mm off its sill
_chord = math.sqrt(tower.R_OUT ** 2 - tower.RODS[tower.SAFE][1] ** 2) + 0.0025   # to the outside of its hole
PULL = tower.SHIFT[tower.SAFE] + tower.ROD_LEN / 2 + _chord + 0.03      # its far end 3 cm past the wall
