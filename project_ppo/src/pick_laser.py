import numpy as np
def Pick(state, batch):
    item = []
    new_state = []
    for elem in state:
        item.append(elem)
        if len(item) == batch:
            new_state.append(np.min(np.asarray(item)))
            item = []
    if len(elem) != 0:
