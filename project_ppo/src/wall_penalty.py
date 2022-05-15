import numpy as np
import math
def pen_wall(state_norm):
    max_state = np.max(np.asarray(state_norm))
    min_state = np.min(np.asarray(state_norm))
    if len(state_norm) % 2 != 0:
        # idx = np.where(np.asarray(state_norm) == np.asarray(state_norm).max())[0]
        idx_middle = len(state_norm) // 2
        value_middle = state_norm[idx_middle]
        # v_mid = value_middle if value_middle > .1 else .1
        # if (max_state - value_middle) > 0.2:
        #     rate_penalty = (max_state - value_middle)
        # else:
        #     rate_penalty = 0
        if min_state <= 0.2:
            rate_penalty = (1 - min_state)
        else:
            rate_penalty = 0
    else:
        idx_middle_g = len(state_norm) // 2
        idx_middle_l = idx_middle_g - 1
        value_middle = max(state_norm[idx_middle_g], state_norm[idx_middle_l])
        # v_mid = value_middle if value_middle > .1 else .1
        # if (max_state - value_middle) > 0.2:
        #     rate_penalty = (max_state - value_middle)
        # else:
        #     rate_penalty = 0
        if min_state <= 0.2:
            rate_penalty = (1 - min_state)
        else:
            rate_penalty = 0
    return rate_penalty


