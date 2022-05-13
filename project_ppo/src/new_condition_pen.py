import numpy as np
state = [1, 1, 3, 0, 2, 0, 5, 5, 1, 0, 0, 8, 6, 5, 7, 8, 0, 7, 1, 1]
diff_angle = 80
def find_dis_from_target(state, diff_angle):
    laser_angles = []
    for i in range(len(state)):
        laser_angles.append(i / (len(state) - 1) * 180)
    right_angles = []
    idx = []
    distance = []
    # if -90 <= diff_angle <= 90:
    for i, elem in enumerate(laser_angles):
        if elem > (90 - diff_angle):
            idx.append(i - 1)
            idx.append(i)
            right_angles.append(laser_angles[i - 1])
            right_angles.append(elem)
            distance.append(state[i - 1])
            distance.append(state[i])
            break
        else:
            pass


print('a')
