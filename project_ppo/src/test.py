

import numpy as np
state = [0.5624945504324776, 0.5786835125514439, 0.626621927533831, 1.0, 1.0, 1.0, 0.6261183193751744, 0.5871847697666713
    , 0.5715985979352679, 0.5577263491494315, 0.5814286640712193, 0.6081752777099609, 0.6661924634660993, 1.0, 1.0, 0.644031115940639
    , 0.5981666701180595, 0.5647472994668143, 0.5582539694649833, 1.0]
def list_angular(state):
    idx = np.where(np.asarray(state)==np.asarray(state).max())[0]
    list = []
    i = 0
    j = 0
    inner_list = []
    while True:
        if j < (np.shape(idx)[0] - 1):
            if idx[j] == idx[j + 1] - 1:
                inner_list.append((1 + idx[j]) / 20 * 180)
            else:
                inner_list.append((1 + idx[j]) / 20 * 180)
                list.append(inner_list)
                inner_list = []
        else:
            inner_list.append((1 + idx[j]) / 20 * 180)
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
list_ang = list_angular(state)
diff_angle = -10
dif_angulars = []
for ang in list_ang:

    dif_angs = (90 - diff_angle) - ang
    if dif_angs < 0:
        dif_angs *= -1
    dif_angulars.append(dif_angs)

sum = 0
alpha = 0.5
dif_angulars.sort()
for i, diffs in enumerate(dif_angulars):
    sum = sum + alpha**i * (90 - diffs)
print('a')
