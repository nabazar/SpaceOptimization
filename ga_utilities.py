from copy import deepcopy
import numpy as np
from math import ceil, hypot
from random import randint, seed
import pdb

from Generator import choose_from_weighted_pool
from plotting_utilities import check_intersections,plot_result

EPS = 1e-9
INF = 1e90
MAX_SEED = int(1e9)
SEED = 43
NUM_POP = 150 #40# 142 # TODO: change it back
GEN_SIZE = 10 #40# 100 # TODO: change it back
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.0592

COLOR1 = 'b'
COLOR2 = 'r'
WIRE_COLOR1 = 'k'
WIRE_COLOR2 = 'm'
WIRE_COLOR3 = 'c'

#SEED = 43

import pdb

#
# whs widths of the components ={13: (3.0, 3.6), 14: (3.9, 4.8), 9: (9.7, 9.7)}
# cuts [0, 1]
# edges if any exist else {}
# perm [13, 14, 9]
# Margin 1
# left_bound
# right_bound
# Algorithm: Checks if any of the components can be kept at 0,0 and then checks based on the rotation if any of the component clashes.
# returns pts which is the xy cordinates of the component where the components can be placed. 
#
def build_tree(left_bound, right_bound, whs, cuts, edges, perm, margin):
    cnt = right_bound - left_bound + 1
    mid = (left_bound + right_bound - 1) // 2
    
    if cnt == 1:
        w, h = whs[perm[left_bound]]
        return {perm[left_bound]: (0, 0)}, w, h, edges.get(perm[left_bound], 0)

    pts1, w1, h1, edge1 = build_tree(left_bound, mid, whs, cuts, edges, perm, margin)
    #print('pts1' , pts1, w1, h1, edge1)
    pts2, w2, h2, edge2 = build_tree(mid + 1, right_bound, whs, cuts, edges, perm, margin)
    #print('pts2' , pts2, w2, h2, edge2)
    pts = deepcopy(pts1)

    #if perm == [3, 8, 6, 7, 2, 5, 1, 0, 4]:
    #    import pdb; pdb.set_trace()
    edge = edge1 | edge2
    if cuts[mid - 1] == 0: # vertical cut
        w = w1 + w2 + margin
        h = max(h1, h2)
        for i, p in pts2.items():
            pts[i] = (p[0] + w1 + margin, p[1])
        if edge1 == -1 or edge2 == -1 or (edge1 & (1 << 1) > 0)\
           or (edge2 & (1 << 3) > 0):
            edge = -1
        for cur_h, cur_pts, cur_edge in zip((h1, h2), (pts1, pts2), (edge1, edge2)):
            if cur_h < h:
                if (cur_edge & (1 << 0) > 0) and (cur_edge & (1 << 2) > 0):
                    edge = -1
                elif cur_edge & (1 << 2) > 0:
                    #print("This is our case!")
                    dh = abs(cur_h - h)
                    for i in cur_pts:
                        p = pts[i]
                        pts[i] = (p[0], p[1] + dh)
    else: # horizontal cut
        h = h1 + h2 + margin
        w = max(w1, w2)
        for i, p in pts2.items():
            pts[i] = (p[0], p[1] + h1 + margin)
        if edge1 == -1 or edge2 == -1 or (edge1 & (1 << 2) > 0)\
           or (edge2 & (1 << 0) > 0):
            edge = -1
        for cur_w, cur_pts, cur_edge in zip((w1, w2), (pts1, pts2), (edge1, edge2)):
            if cur_w < w:
                if (cur_edge & (1 << 1) > 0) and (cur_edge & (1 << 3) > 0):
                    edge = -1
                elif cur_edge & (1 << 1) > 0:
                    #print("This is our horizontal case!")
                    dw = abs(cur_w - w)
                    for i in cur_pts:
                        p = pts[i]
                        pts[i] = (p[0] + dw, p[1])
    #print(pts, w, h, edge, left_bound, right_bound, whs, cuts, edges, perm, margin)
    return pts, w, h, edge

#
# Calculates the hypotunues of the shape so that the maximum area can be found.
#
def calc_base(shapes):
    s = 0
    for p in shapes:
        shape = shapes[p]
        s += hypot(shape[1], shape[0])
    return s * len(shapes)
#
# calculates the fitness function
#
def calc_fitness_function(individual, shapes, edges,
                          connections, MARGINS, conn_place):
    #if individual == ([35, 16, 70], [1, 1], [3, 2, 3], [], [], []):
    #    import pdb; pdb.set_trace()
    result = 0
    perm1, cuts1, rots1, perm2, cuts2, rots2 = individual
    w1, h1, pts1, whs1, final_conn_place1, edge1 = calc_rects(perm1, cuts1, rots1,
                                               shapes, edges, conn_place, MARGINS[0])
    #print('\nw1 ',w1, '\nh1 ',h1, '\npts1 ',pts1, '\nwhs1 ',whs1,'\nedge1 ', edge1,'\nindividual', perm1, cuts1, rots1 )

    w2, h2, pts2, whs2, final_conn_place2, edge2 = calc_rects(perm2, cuts2, rots2,
                                               shapes, edges, conn_place, MARGINS[1])
    #print('\nw2 ',w2, '\nh2 ',h2, '\npts2 ',pts2, '\nwhs2 ',whs2,'\nedge2 ', edge2,'\nindividual', perm2, cuts2, rots2 )

    final_conn_place = deepcopy(final_conn_place1)
    final_conn_place.update(final_conn_place2)

    best_area = INF
    best_wire_length = INF
    pts, whs, colors = None, None, None
    best_wires = None
    best_w = None
    best_h = None

    w = max(w1, w2)
    h = max(h1, h2)

    #if ([49], [], [2], [48], [], [1]) == individual:
    #    import pdb; pdb.set_trace()
    #import pdb; pdb.set_trace()
    #if individual == ([2, 1, 0], [1, 0], [0, 0, 3], [], [], []):
    #    import pdb; pdb.set_trace()

    side1_wires_length, wires1 = calc_one_side_wire_length(perm1, pts1, whs1,
                                    connections, WIRE_COLOR1, final_conn_place)
    side2_wires_length, wires2 = calc_one_side_wire_length(perm2, pts2, whs2,
                                    connections, WIRE_COLOR2, final_conn_place)

    for wires in [wires1, wires2]:
        for wire in wires:
            x1, y1, x2, y2, _ = wire

            if max(x1, x2) > w:
                continue
                import pdb; pdb.set_trace()
                
            if max(y1, y2) > h:
                continue
                import pdb; pdb.set_trace()

    between_sides_wires_length, wires3 = \
        calc_wire_length_connecting_different_sides(w, h, perm1, perm2,
                                            pts1, pts2,
                                            connections, final_conn_place)

                
    current_wire_length = side1_wires_length + \
        side2_wires_length + between_sides_wires_length
    if w * h < best_area or w * h == best_area and \
        current_wire_length < best_wire_length:
        best_area = w * h
        best_wire_length = current_wire_length
        best_wires = wires1 + wires2 + wires3
        whs = deepcopy(whs1)
        whs.update(whs2)
        pts = deepcopy(pts1)
        pts.update(pts2)
        best_w = w
        best_h = h

    colors = {x: COLOR1 for x in perm1}
    colors.update({x: COLOR2 for x in perm2})

    rotations = {p: r for p, r in zip(perm1, rots1)}
    rotations.update({p: r for p, r in zip(perm2, rots2)})
    # area, wire_length, w, h, pts, whs, colors, wires

    for i, pin in final_conn_place:
        x, y = final_conn_place[(i, pin)]
        '''
        if x > best_w + pts[i][0] or y > best_h + pts[i][1]:
            print("Right, Sherlock")
            import pdb; pdb.set_trace()

        '''
    return best_area, best_wire_length, best_w, best_h, pts, whs, \
            colors, best_wires, rotations, final_conn_place, edge1, edge2

def calc_nodes_numbers(n):
    nodes_numbers = [0] * (n + 1)
    
    for i in range(2, n + 1):
        left_bound = 0
        right_bound = i - 1
        mid = (left_bound + right_bound - 1) // 2
        nodes_numbers[i] = 1 + nodes_numbers[mid - left_bound + 1] +\
            nodes_numbers[right_bound - (mid + 1) + 1]
    return nodes_numbers

def calc_one_side_wire_length(perm, pts, whs, connections, color, final_conn_place):
    current_board = set(perm)
    wires = []
    total_length = 0

    #if {2: (0, 0), 1: (4.3, 0), 0: (4.3, 19.9)} == pts:
    #    import pdb; pdb.set_trace()
    pin = [x[0][1] for x in connections]#[0]
    #if len(pin) > 0 and isinstance(pin[0], tuple):
    #    import pdb; pdb.set_trace()

    for point1, point2 in connections:
        footprint1, pin1 = point1
        footprint2, pin2 = point2
        if footprint1 not in current_board or \
           footprint2 not in current_board:
            continue
        x1, y1 = pts[footprint1]
        x2, y2 = pts[footprint2]
        px1, py1 = final_conn_place[point1]
        px2, py2 = final_conn_place[point2]

        for px, py, footprint in zip((px1, px2), (py1, py2), (footprint1, footprint2)):
            if px < pts[footprint][0] or px > pts[footprint][0] + whs[footprint][0]:
                print("px out of range!")
                import pdb; pdb.set_trace()
            if py < pts[footprint][1] or py > pts[footprint][1] + whs[footprint][1]:
                print("py out of range!")
                import pdb; pdb.set_trace()

        wires.append((px1, py1, px2, py2, color))
        total_length += hypot(py1 - py2, px1 - px2)

    return total_length, wires
    
def calc_wire_length_connecting_different_sides(w, h, perm1, perm2,
                                                pts1, pts2, connections,
                                                final_conn_place):
    top_board = set(perm1)
    bottom_board = set(perm2)
    all_wires = []
    total_length = 0
    #import pdb; pdb.set_trace()
    for point1, point2 in connections:
        footprint1, pin1 = point1
        footprint2, pin2 = point2
        if not (footprint1 in top_board and footprint2 in bottom_board or \
           footprint1 in bottom_board and footprint2 in top_board):
            continue
        #import pdb; pdb.set_trace()
        best_len = INF
        best_wires = None

        if footprint1 in top_board:
            x1, y1 = pts1[footprint1]
            x2, y2 = pts2[footprint2]
        else:
            x1, y1 = pts1[footprint2]
            x2, y2 = pts2[footprint1]

        v1 = ((1, 0), (-1, 2 * h))
        v2 = ((-1, 2 * w), (1, 0))
        v3 = ((1, 0), (-1, 0))
        v4 = ((-1, 0), (1, 0))

        vs = [v1, v2, v3, v4]
        lines = [(1, h), (0, w), (1, 0), (0, 0)]
        for v, line in zip(vs, lines):
            ax, bx = v[0]
            ay, by = v[1]
            nx2 = ax * x2 + bx
            ny2 = ay * y2 + by
            
            
            length = hypot(ny2 - y1, nx2 - x1)
            if length >= best_len:
                continue

            best_len = length

            a = ny2 - y1
            b = x1 - nx2
            c = -a * x1 -b * y1

            line_type, line_c = line
            if line_type == 1: # horizontal line
                if abs(a) < EPS:
                    best_wires = [(x1, y1, x2, y2, WIRE_COLOR3)]
                    continue
                y0 = line_c
                x0 = -(.0 + b * y0 + c) / a

                best_wires = [(x1, y1, x0, y0, WIRE_COLOR1),
                              (x2, y2, x0, y0, WIRE_COLOR2)]
            else: # vertical line
                if abs(b) < EPS:
                    best_wires = [(x1, y1, x2, y2, WIRE_COLOR3)]
                    continue
                x0 = line_c
                y0 = -(.0 + a * x0 + c) / b

                best_wires = [(x1, y1, x0, y0, WIRE_COLOR1),
                              (x2, y2, x0, y0, WIRE_COLOR2)]
        
        total_length += best_len
        all_wires.extend(best_wires)
    return total_length, all_wires

#
# Generates the width, height, new connections based on the rotation set.
# Generates pts which contains the position of the components to be used. 
#
def calc_rects(perm, cuts, rots, shapes, edges, conn_place, margin):
    if perm == []:
        return 0, 0, {}, {}, {}, 0
    inv_perm = {p: i for i, p in enumerate(perm)}
    whs = {}
    new_conn_place = {}

    #if shapes == {0: (16.7, 21.599999999999998), 1: (30.96, 21.8), 2: (38.4, 18.64), 3: (35.14, 18.64), 4: (20.66, 16.44), 5: (14.48, 19.83), 6: (15.8, 15.2), 7: (25.599999999999998, 5.2), 8: (9.2, 11.2)}:
    #    import pdb; pdb.set_trace()
    #import pdb; pdb.set_trace()
    for i, pin in conn_place:
        if i not in inv_perm: # this shape isn't used on that side of board
            continue
        dx, dy = conn_place[(i, pin)]
        r = rots[inv_perm[i]]
        w, h = shapes[i]
        
        if r == 0:
            new_conn_place[(i, pin)] = dx, dy
        elif r == 1:
            new_conn_place[(i, pin)] = h - dy, dx
        elif r == 2:
            new_conn_place[(i, pin)] = w - dx, h - dy
        elif r == 3:
            new_conn_place[(i, pin)] = dy, w - dx

        if new_conn_place[(i, pin)][1] < 0:
            import pdb; pdb.set_trace()

    new_edges = {}
    for num, mask in edges.items():
        for i in range(4):
            if mask & (1 << i) > 0:
                new_i = (i + rots[inv_perm[num]]) % 4
                if num not in new_edges:
                    new_edges[num] = 0
                new_edges[num] += 1 << new_i
    for i, p in enumerate(perm):
        w, h = shapes[p]

        if rots[i] % 2:
            w, h = h, w
        whs[p] = (w, h)

    nodes_number = calc_nodes_numbers(len(perm))
    pts, W, H, edge = build_tree(0, len(perm) - 1, whs, cuts, new_edges, perm, margin)
    final_conn_place = {}
    
    for i, pin in new_conn_place:
        if i not in inv_perm:
            import pdb; pdb.set_trace()
            continue
        dx, dy = new_conn_place[(i, pin)]

        nx, ny = pts[i][0] + dx, pts[i][1] + dy
        
        w, h = whs[i]
        if nx < pts[i][0] or nx > pts[i][0] + w or ny < pts[i][1] or ny > pts[i][1] + h:
            print('error seen as point outside the boundary', pin)
            #import pdb; pdb.set_trace()
        final_conn_place[(i, pin)] = nx, ny
   
    return W, H, pts, whs, final_conn_place, edge

# performs crossover of xp1 part of chromosome, which
# is permutation of shapes for top and bottom
def crossover_xp1(g1, g2, g3, g4): # top1, bottom1, top2, bottom2
    crossover_point1 = 0
    if len(g1) > 0:
        crossover_point1 = randint(1, len(g1))
    crossover_point2 = 0
    if len(g2) > 0:
        crossover_point2 = randint(1, len(g2))
    top = g1[:crossover_point1]
    used_genes = set(top)
    whole_perm2 = g3 + g4
    for g in whole_perm2:
        if len(top) == len(g1):
            break
        if g not in used_genes:
            top.append(g)
            used_genes.add(g)
    bottom = []
    
    for g in g2:
        if len(bottom) == crossover_point2:
            break
        if g in used_genes:
            continue
        bottom.append(g)
        used_genes.add(g)
    whole_perm2 = g4 + g3
    for g in whole_perm2:
        if len(bottom) == len(g2):
            break
        if g not in used_genes:
            bottom.append(g)
    return top, bottom
    
# performs crossover of xp2 part of chromosome, which
# describes types of each cut (0 - vertical, 1 - horizontal)
def crossover_xp2(g1, g2, g3, g4):
    crossover_point1 = 0
    if len(g1) > 0:
        crossover_point1 = randint(1, len(g1))
    crossover_point2 = 0
    if len(g2) > 0:
        crossover_point2 = randint(1, len(g2))
    top = deepcopy(g1)
    bottom = deepcopy(g2)

    for i in range(crossover_point1, min(len(g1), len(g3) + crossover_point1)):
        top[i] = g3[i - crossover_point1]

    for i in range(crossover_point2, min(len(g2), len(g4) + crossover_point2)):
        bottom[i] = g4[i - crossover_point2]

    #print("XP2:", len(g1), len(g2), len(top), len(bottom))
    return top, bottom
     
# performs crossover of xp3 part of chromosome, which
# describes rotation of each shape (0 - vertical, 1 - horizontal)
def crossover_xp3(g1, g2, g3, g4):
    #print("It's XP3, not XP2")
    return crossover_xp2(g1, g2, g3, g4)
       
def crossover(ind1, ind2, random_seed):
    seed(random_seed)
    xp11, xp12 = crossover_xp1(ind1[0], ind1[3], ind2[0], ind2[3])
    xp21, xp22 = crossover_xp2(ind1[1], ind1[4], ind2[1], ind2[4])
    xp31, xp32 = crossover_xp3(ind1[2], ind1[5], ind2[2], ind2[5])

    random_seed = randint(0, MAX_SEED)
    return (xp11, xp21, xp31, xp12, xp22, xp32), random_seed

def find_connection_between_two_shapes(pts1, whs1, pts2, whs2):
    x1, y1 = pts1
    w1, h1 = whs1
    x2 = x1 + w1
    y2 = y1 + h1
        
    x3, y3 = pts2
    w2, h2 = whs2
    x4 = x3 + w2
    y4 = y3 + h2

    xmax = max(x1, x3)
    xmin = min(x2, x4)

    ymax = max(y1, y3)
    ymin = min(y2, y4)

    best_length = INF
    bx1, by1, bx2, by2 = None, None, None, None
    if xmax <= xmin:
        py1 = y2
        py2 = y3

        
        if y1 > y3:
            py1 = y1
            py2 = y4

        length = abs(py1 - py2)

        if length < best_length:
            best_length = length
            bx1, by1, bx2, by2 = xmax, py1, xmax, py2

    if ymax <= ymin:
        px1 = x2
        px2 = x3

        
        if x1 > x3:
            px1 = x1
            px2 = x4

        length = abs(px1 - px2)

        if length < best_length:
            best_length = length
            bx1, by1, bx2, by2 = px1, ymax, px2, ymax

    px1, px2 = x2, x3
    py1, py2 = y2, y3

    if x1 > x3:
        px1, px2 = x1, x4
    if y1 > y3:
        py1, py2 = y1, y4

    length = hypot(py2 - py1, px2 - px1)

    if length < best_length:
        best_length = length
        bx1, by1, bx2, by2 = px1, py1, px2, py2

    return bx1, by1, bx2, by2

#
# n stands for number of objects which need to be parsed
# shapes contains length and breadth of each component
# connections contains netlistt connections between components. Removes the power connections
# conn_place contains the locations of the pins taken i guess with x y equal to zero
# edges- component edges
# rd_mapping- mapping of refdes and package
# Margins which contains the distance between components
# top side states place only on top side. Value is set to true
# area_constraints True or false
# Gen_size is how many evolutions are needed
# NUM_POP is not sure. Selected is 15
# Seed is the number of the initial population
# GEN_SIZE=40
#
def find_solution(shapes, connections, conn_place, edges, footprint_num,
                  rd_mapping, margins, top_side_only, area_constraint,
                  picture_name, gen_size=GEN_SIZE, num_pop=NUM_POP):
    n = len(shapes)
    seed(SEED)
    random_seed = SEED
    # Find the maximum length of the three objects so that the area can be found.
    #print(shapes)
    base = calc_base(shapes)

    best_ind = None
    best_fitness_func = INF, INF
    best_pts = None
    best_whs = None
    best_wires = None
    best_w = None
    best_h = None
    best_colors = None
    best_rotations = None
    population = []
    
    for i in range(num_pop):
        # Generates the random seed which has component index , cuts and rotations
        # ([14, 9, 13], [1, 0], [0, 0, 3], [], [], []) 51718274
        individual, random_seed = gen_individual(n, shapes.keys(), top_side_only, random_seed)

        area, wire_length, w, h, pts, whs, colors, wires, _, _, edge1, edge2 = \
                   calc_fitness_function(individual, shapes, edges,
                                         connections, margins, conn_place)

        #print('fitness check',area, wire_length, round(area*wire_length,0), w, h, pts)
        if area < 0 or edge1 < 0 or edge2 < 0:
            continue

        population.append(individual)

    for generation in range(gen_size):
        #print("Best area is {1} " "wire length is {2}".format(generation, *best_fitness_func))
        
        if len(population) < 2:
            print("Terminated because length of population < 2 on "
                  "iteration {0}".format(generation))
        
        fitness = []

        available_inds = []
        for ind in population:
            area, wire_length, w, h, pts, whs, colors, wires,\
                rotations, final_conn_place, edge1, edge2 =\
                  calc_fitness_function(ind, shapes, edges, connections, margins, conn_place)

            if area < 0 or edge1 < 0 or edge2 < 0:
                continue

            if wire_length==0: # inserted by manav to cover for cases where no wires exist for the two interfaces
                wire_length=1
            fitness_func = (area, wire_length)
            if area_constraint:
                fitness_func = (area, wire_length)
            else:
                fitness_func = (wire_length,area)

            fitness_func = (area*wire_length, wire_length)  # changed by manav
            
            if fitness_func < best_fitness_func:
                best_fitness_func = fitness_func
                best_ind = individual
                best_pts = pts
                best_whs = whs
                best_colors = colors
                best_w = w
                best_h = h
                best_wires = wires
                best_rotations = rotations
                best_final_conn_place = final_conn_place
                best_edge1 = edge1
                best_edge2 = edge2
            fitness.append(make_flat(fitness_func, base))
            available_inds.append(ind)
            
        #print('available_inds', available_inds[0])
        #print('Best individual', best_ind)
        #print(' best_pts', best_pts)
        #print(' best_fitness_func', best_fitness_func)
        #print(available_inds,len(available_inds))

        # creating a pool of individuals which will take part in crossover
        pool, random_seed = choose_from_weighted_pool(available_inds, fitness,
                                         len(available_inds) // 2, random_seed)

        if len(pool) < 3:
            print("Pool exhausted, terminating ga algorithm")
            break

        new_population = []
        # crossover
        crossover_count = int(ceil(CROSSOVER_RATE * len(population)))
        result_of_crossover = []
        for i in range(crossover_count):
            c1 = randint(0, len(pool) - 2)
            c2 = randint(c1 + 1, len(pool) - 1)
            new_individual, random_seed = crossover(pool[c1], pool[c2],
                                                    random_seed)
            result_of_crossover.append(new_individual)
            
        # mutation
        mutation_count = int(ceil(MUTATION_RATE * len(population)))
        result_of_mutation = []
        for i in range(mutation_count):
            c = randint(0, len(result_of_crossover) - 1)

            new_individual, random_seed = mutation(result_of_crossover[c],
                                                   random_seed)
            result_of_mutation.append(new_individual)
        population = result_of_crossover + result_of_mutation

        #print(population)
        #input()
    '''
    with open("result.txt", "w") as f:
        print("Rectangles:")
        f.write("Rectangles:\n")
        for p in best_pts:
            pt = best_pts[p]
            wh = best_whs[p]
            color = best_colors[p]
            rot = best_rotations[p] * 90
            print("Point ({0}, {1}) with width = {2} height = {3} rotated on {4} has color {5}".\
                  format(*pt, *wh, rot, color))
            f.write("Point ({0}, {1}) with width = {2} height = {3} rotated on {4} has color {5}\n".\
                  format(*pt, *wh, rot, color))

        print("Wires:")
        f.write("Wires:\n")
        for x1, y1, x2, y2, color in best_wires:
            print(x1, y1, x2, y2, color)
            f.write("Starts at ({0}, {1}) and finishes at ({2}, {3})\n".\
                    format(x1, y1, x2, y2))
    '''

    #print("Area: {0}".format(best_w * best_h))
    #print("w = {0}, h = {1}".format(best_w, best_h))
    #print("Best rotations", best_rotations)
    check_intersections(best_pts, best_whs, best_colors)

    #if picture_name == "global.jpg":
    #    import pdb; pdb.set_trace()
    
    #save_to_xml(best_pts, best_whs, best_colors, best_rotations,footprint_num, rd_mapping)
    #plot_result(best_pts, best_whs, best_w, best_h, best_colors, best_wires, picture_name)

    """
    subdata = formate_data(best_pts, best_whs, best_colors,
                          best_rotations, footprint_num, rd_mapping)
    
    
    res = calc_fitness_function(
                best_ind, shapes, edges, connections, margins, conn_place)
    final_conn_place, edge1, edge2 = res[-3:]
    import pdb; pdb.set_trace()
    """    
    return best_w, best_h, best_final_conn_place, best_edge1, best_edge2,\
            best_rotations, best_pts, best_whs, best_colors, best_wires, footprint_num

# Returns individual as ([9, 13, 14], [0, 0], [0, 2, 0], [], [], [])
#                       ([13, 14, 9], [1, 1], [0, 2, 1], [], [], [])
# and an number which is nine digit 471146370
def gen_individual(n, keys, TOP_SIDE_ONLY, random_seed):
    np.random.seed(random_seed)
    seed(random_seed)
    if TOP_SIDE_ONLY:
        individual = gen_xp1(n, keys), gen_xp2(n), gen_xp3(n), \
                         [], [], []
    else:
        perm = gen_xp1(n, keys)
        n1 = randint(1, n - 1)
        n2 = n - n1
        xp11 = perm[:n1]
        xp12 = perm[n1:]
        individual = xp11, gen_xp2(n1), gen_xp3(n1), \
                         xp12, gen_xp2(n2), gen_xp3(n2)

    random_seed = randint(0, MAX_SEED)
    return individual, random_seed

# Generates the random order of the component index or keys
# [9, 14, 13]
def gen_xp1(n, keys):
    if n == 0:
        return []
    l = list(keys)
    perm = np.random.permutation(n)
    perm = perm.tolist()
    result = [l[i] for i in perm]
    return result

# Generate random order such as [1, 0] or [1, 1] or [0, 1]
def gen_xp2(n):
    if n == 0:
        return []
    mask = []
    for i in range(n - 1):
        bit = randint(0, 1)
        mask.append(bit)
    return mask

# Generate random order such as [3, 0, 0]
def gen_xp3(n):
    if n == 0:
        return []
    mask = []
    for i in range(n):
        bit = randint(0, 3)
        mask.append(bit)
    return mask

def make_flat(l, base):
    return l[0] * base + l[1]

# performs mutation of xp1 part of chromosome, which
# is permutation of shapes
def mutation_xp1(g):
    if len(g) < 2:
        return deepcopy(g)
    result = deepcopy(g)
    c1 = randint(0, len(g) - 2)
    c2 = randint(c1 + 1, len(g) - 1)
    result[c1], result[c2] = result[c2], result[c1]
    return result
    
# performs mutation of xp2 part of chromosome, which
# describes types of each cut (0 - vertical, 1 - horizontal)
def mutation_xp2(g):
    if len(g) == 0:
        return deepcopy(g)

    cnt = randint(1, 3)
    result = deepcopy(g)
    for i in range(cnt):
        mutation_point = randint(0, len(result) - 1)
        result[mutation_point] ^= 1
    return result
    
# performs mutation of xp3 part of chromosome, which
# describes rotations
def mutation_xp3(g):
    if len(g) == 0:
        return deepcopy(g)
    cnt = randint(1, 3)
    result = deepcopy(g)
    for i in range(cnt):
        mutation_point = randint(0, len(result) - 1)
        mutation_value = randint(0, 3)
        result[mutation_point] = mutation_value
    return result
        
def mutation(ind, random_seed):
    # TODO: remove
    seed(random_seed)
    mask = randint(1, 63)
    xp11 = deepcopy(ind[0])
    xp21 = deepcopy(ind[1])
    xp31 = deepcopy(ind[2])
    xp12 = deepcopy(ind[3])
    xp22 = deepcopy(ind[4])
    xp32 = deepcopy(ind[5])
    if mask & 1:
        xp11 = mutation_xp1(ind[0])
    if mask & 2:
        xp21 = mutation_xp2(ind[1])
    if mask & 4:
        xp31 = mutation_xp3(ind[2])
    if mask & 8:
        xp12 = mutation_xp1(ind[3])
    if mask & 16:
        xp22 = mutation_xp2(ind[4])
    if mask & 32:
        xp32 = mutation_xp3(ind[5])

    new_random_seed = randint(0, MAX_SEED)
    return (xp11, xp21, xp31, xp12, xp22, xp32), new_random_seed

def permute_connections(perm1, perm2, connections, con_place):
    inv_perm1 = {x : i for i, x in enumerate(perm1)}
    inv_perm2 = {x : i for i, x in enumerate(perm2)}
    permuted_connections = set()
    for p in connections:
        t1, t2 = p
        i, pin1 = t1
        j, pin2 = t2
        permuted_connections.add(((inv_perm1[i], pin1), (inv_perm2[j], pin2)))
    new_con_place = {}
    for t, dxdy in con_place:
        i, pin = t
        new_i = None
        if i in inv_perm1:
            new_i = inv_perm1[i]
        else:
            new_i = inv_perm2[i]
        if new_i is None:
            raise Exception("ga_utilities.permute_connections: Haven't found thing in permutations")
        new_con_place[(new_i, pin)] = t
    
    return permuted_connections, new_con_place
