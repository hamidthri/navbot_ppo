import numpy as np
def Pick(state, len_batch):
    item = []
    new_state = []
    for elem in state:
        item.append(elem)
        if len(item) == len_batch:
            new_state.append(np.min(np.asarray(item)))
            item = []
    if len(item) != 0:
        new_state.append(np.min(np.asarray(item)))
    return new_state
