import numpy as np


state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
state_angle = []
for i in range(len(state)):
    state_angle.append(i / (len(state) - 1) * 180)

dif_angle = 15
angles = []
index = []
for i in range(len(state)):
    if 90 - dif_angle > state_angle[i]:
        pass
    elif 90 - dif_angle == state_angle[i]:
        angles.append(state_angle[i])
        index.append(i)
        break
    else:
        angles.append(state_angle[i - 1])
        index.append(i - 1)
        index.append(i)

        angles.append(state_angle[i])
        break
avg = 0
for i in range(len(index)):
    avg = avg + state[index[i]]
avg = avg / len(index)
print('a')