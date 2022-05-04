import math

def reletiveangle(rel_dis_x, rel_dis_y):
    # Calculate the angle between robot and target
    if rel_dis_x > 0 and rel_dis_y > 0:
        theta = math.atan(rel_dis_y / rel_dis_x)
    elif rel_dis_x > 0 and rel_dis_y < 0:
        theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
    elif rel_dis_x < 0 and rel_dis_y < 0:
        theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
    elif rel_dis_x < 0 and rel_dis_y > 0:
        theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
    elif rel_dis_x == 0 and rel_dis_y > 0:
        theta = 1 / 2 * math.pi
    elif rel_dis_x == 0 and rel_dis_y < 0:
        theta = 3 / 2 * math.pi
    elif rel_dis_y == 0 and rel_dis_x > 0:
        theta = 0
    else:
        theta = math.pi
    rel_theta = round(math.degrees(theta), 2)
    return rel_theta
