

import numpy as np
state = [1, 1, 1, 0, 1, 0, 5, 5, 6, 1, 1, 8, 6, 5, 7, 8, 1, 7, 1, 1]
def list_angular(state, num_scan):
    idx = np.where(np.asarray(state)==np.asarray(state).max())[0]
    idx = np.where(np.asarray(state) == 1.0)[0]
    list = []
    j = 0
    inner_list = []
    while True:
        if np.shape(idx)[0] == 0:
            break
        if j < (np.shape(idx)[0] - 1):
            if idx[j] == idx[j + 1] - 1:
                inner_list.append((idx[j]) / (num_scan - 1) * 180)
            else:
                inner_list.append((idx[j]) / (num_scan - 1) * 180)
                list.append(inner_list)
                inner_list = []
        else:
            inner_list.append((idx[j]) / (num_scan - 1) * 180)
            list.append(inner_list)
            break
        j += 1
    list_ang = []
    for i in range(len(list)):
        sum = 0
        for k, j in enumerate(list[i]):
            sum += j
        sum /= k + 1
        list_ang.append(sum)
    return list_ang
list_ang = list_angular(state, num_scan=20)
if len(list_ang) != 0:
    diff_angle = 10
    dif_angulars = []
    for ang in list_ang:

        dif_angs = (90 - diff_angle) - ang
        if dif_angs < 0:
            dif_angs *= -1
        dif_angulars.append(dif_angs)

    suma = 0
    alpha = 0.5
    dif_angulars.sort()
    for i, diffs in enumerate(dif_angulars):
        suma = suma + alpha**i * (90 - diffs)
print('a')
