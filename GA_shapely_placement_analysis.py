'''
program created which creates a solution with GA using shapely obects. Uses DEAP library. 
This program works best when the number of objects is less in number.
'''

import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import rotate , translate
import shapely.geometry as geom
from shapely.ops import unary_union
from deap import base, creator, tools, algorithms
import random
from multiprocessing import Pool
import time
import pickle

####################################################################
# Transform and rotation function
####################################################################
def Find_wire_distance(individual,polygons,connection,conn_place,Fixed_Polygons,Transform_and_Rotate_dict):
    Wires = []
    Wire_labels = []
    total_distance = 0
    for poly1_conn, poly2_conn in connection:
        ''' Point1 transformation '''
        poly1_Indx = poly1_conn[0]
        poly1_point1 = geom.Point(conn_place[poly1_conn])
        #print(poly1_conn, poly1_Indx , poly1_point1)
        if Transform_and_Rotate_dict[poly1_Indx] == [0,0,0,(0,0)]:
            temp_poly_point1 = poly1_point1
        else:
            [Individual1, Individual2 , angle, bottom_left] = Transform_and_Rotate_dict[poly1_Indx]
            #print(poly1_conn, poly1_point1, Individual1 , Individual2 , angle, bottom_left)
            temp_poly_point1 = translate(poly1_point1, Individual1, Individual2)
            temp_poly_point1 = rotate(temp_poly_point1, angle, origin=bottom_left)

        ''' Point2 transformation '''
        poly2_Indx = poly2_conn[0]
        poly2_point1 = geom.Point(conn_place[poly2_conn])        
        if Transform_and_Rotate_dict[poly2_Indx] == [0,0,0,(0,0)]:
            temp_poly_point2 = poly2_point1
        else:
            [Individual1, Individual2 , angle, bottom_left] = Transform_and_Rotate_dict[poly2_Indx]
            temp_poly_point2 = translate(poly2_point1, Individual1, Individual2)
            temp_poly_point2 = rotate(temp_poly_point2, angle, origin=bottom_left)
        Wires.append([temp_poly_point1, temp_poly_point2])
        Wire_labels.append([poly1_conn, poly2_conn])
        total_distance += temp_poly_point1.distance(temp_poly_point2)

    return(total_distance, Wires, Wire_labels)
####################################################################
# Transform and rotation function
####################################################################
def Transform_and_Rotate_Shapes(individual,polygons,connection,conn_place,Fixed_Polygons):    
    transformed_polygons = []  
    ''' Contains [x_diff, y_diff, Rotation and left corner(x,y)] '''
    Transform_and_Rotate_dict = {} 
    for i in range(len(polygons)):
        if i in Fixed_Polygons:
            transformed_polygons.append(polygons[i])
            Transform_and_Rotate_dict[i] = [0,0,0,(0,0)]  
            continue
        # Translate then rotate the polygon
        idx = 3*(i - (len(Fixed_Polygons)))
        angle = round(individual[idx + 2],0) * 90
        temp_polygon = translate(polygons[i], individual[idx], individual[idx + 1])
        minx, miny, maxx, maxy = temp_polygon.bounds
        bottom_left = (minx, miny)
        temp_polygon = rotate(temp_polygon, angle, origin=bottom_left)
        transformed_polygons.append(temp_polygon)
        Transform_and_Rotate_dict[i] = [individual[idx], individual[idx + 1] , angle, bottom_left]
    return(transformed_polygons,Transform_and_Rotate_dict)

####################################################################
# Find the area
####################################################################
def find_polygon_area(transformed_polygons):
    combined = unary_union(transformed_polygons)
    minx, miny, maxx, maxy = combined.bounds
    Grp_width = maxx - minx
    Grp_height = maxy - miny
    Area = Grp_width * Grp_height
    return(Area)


####################################################################
# Custom mutation function to handle the rotation gene and others
####################################################################
def evalPolygons_with_SomeFixed(polygons,connection,conn_place,Fixed_Polygons,individual):
    penalty = 0
    total_distance = 0 
    #total_distance, transformed_polygons , _ , _  = Transform_and_Rotate_Shapes(individual,polygons,connection,conn_place,Fixed_Polygons)
    transformed_polygons,Transform_and_Rotate_dict = Transform_and_Rotate_Shapes(individual,polygons,connection,conn_place,Fixed_Polygons)
    for i, polygon in enumerate(transformed_polygons):       
        # Check for overlap and minimum distance with other polygons
        for j, other_polygon in enumerate(transformed_polygons):
            if i >= j:  # Avoid checking a polygon with itself and re-checking pairs
                continue
            if polygon.intersects(other_polygon):
                penalty += float(100000000)
                break
            elif polygon.distance(other_polygon) < 1:
                penalty += float(100000000)
                break
    if penalty <100000000 :
        total_distance, _, _ = Find_wire_distance(individual,polygons,connection,conn_place,Fixed_Polygons,Transform_and_Rotate_dict)
    #   total_area = sum(polygon.area for polygon in transformed_polygons)
    total_area = find_polygon_area(transformed_polygons)
    T = (total_distance + penalty)* total_area
    #print(T)
    return (T),

####################################################################
# Custom mutation function to handle the rotation gene and others
####################################################################
def mutate(MUTPB,individual):
    #print(MUTPB,individual)
    for i in range(0, len(individual), 3):  # Iterate through each group of parameters
        if random.random() < MUTPB:
            # Apply mutation to the x and y offsets
            individual[i] += random.uniform(-1, 1)  # Mutate x offset
            individual[i+1] += random.uniform(-1, 1)  # Mutate y offset
            # Ensure the rotation index stays within 0-3 range
            #individual[i+2] = int((individual[i+2] + random.randint(0, 3)) % 4)
            rotation_mutation = random.choice([-1, 0, 1])
            # Apply mutation step and ensure result is within 0-3 range
            individual[i+2] = (int(individual[i+2]) + rotation_mutation) % 4
    
    return individual,

##################################################################################
# GA function to do the analysis
##################################################################################
def GA_shapely_placement_analysis_stage3(brd_path, MultiPoly_Object_List, global_connections, global_conn_place, global_margins):
    t1 = time.time()
    ''' Here global_shapes is a multipolygon shape '''
    Debug = False
    if Debug : 
        for i, multi_polygon in enumerate(MultiPoly_Object_List):
            print(multi_polygon)

            fig, ax2 = plt.subplots()
            ax2.set_aspect('equal')

            for polygon in multi_polygon.geoms:
                # Exterior coordinates split into x's and y's
                x, y = polygon.exterior.xy
                ax2.fill(x, y, alpha=0.5)  # Fill the polygon with a semi-transparent color

            for key, pnt in global_conn_place.items():
                if i == key[0]:
                    ax2.plot(pnt[0], pnt[1], 'bo')
            plt.show()

    Fixed_Polygons = [0]

    No_of_Solutions = len(MultiPoly_Object_List)-len(Fixed_Polygons)
    #print('No of solutions', No_of_Solutions)

    # Genetic Algorithm parameters
    # New experimental values
    Multiple = 1
    NGEN = No_of_Solutions * Multiple
    CXPB, MUTPB ,ELITISM = 0.75, 0.2 , 0.1
    Population = No_of_Solutions * 150
    NGEN = 5
    Population = 100
    print(f'No of Generations {NGEN} & Pop {Population}')

   
    # Setup the problem
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -250, 250)
    toolbox.register("attr_rotation", random.randint, 0, 3)
    # Individual creation now includes three attributes: x offset, y offset, and rotation
    toolbox.register("individual", tools.initCycle, creator.Individual,(toolbox.attr_float, toolbox.attr_float, toolbox.attr_rotation), n=No_of_Solutions)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mutate", mutate, MUTPB)
    toolbox.register("evaluate",evalPolygons_with_SomeFixed, MultiPoly_Object_List , global_connections , global_conn_place,Fixed_Polygons)
    toolbox.register("mate", tools.cxBlend, alpha=0.5) # Crossover
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population
    population = toolbox.population(n=Population)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    #pool = Pool()
    #toolbox.register("map", pool.map)

    # Initialize the best fitness value and counter for generations without improvement
    best_fitness = float("inf")
    generations_without_improvement = 0
    num_elite = int(ELITISM * len(population))
    for gen in range(NGEN):
        # Select elite individuals
        elites = tools.selBest(population, num_elite)
        # Ensure elite individuals are not lost during cloning for crossover/mutation
        elites = list(map(toolbox.clone, elites))
        
        # All the evolutionary process except elitism is handled by eaSimple
        offspring = algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=1, verbose=False)[0]

        # Combine elites with the rest of the offspring
        # Elites are directly carried over to the next generation
        offspring.extend(elites)
        
        # Update the population
        population[:] = offspring
       
        best_ind = tools.selBest(population, 1)[0]
        print(f"{gen}:{best_ind.fitness.values}")
        # Check for improvement
        current_best = round(best_ind.fitness.values[0],0)
        if current_best < best_fitness - 2000 :
            best_fitness = current_best
            generations_without_improvement = 0  # Reset counter
        else:
            generations_without_improvement += 1  # Increment counter
        
        # Stop if no improvement for more than 20 generations
        if generations_without_improvement > 20:
            print(f"Stopping")
            break
    
    # Run the GA
    #final_population = algorithms.eaSimple(population, toolbox, CXPB, MUTPB, NGEN, verbose=False)[0]

    #pool.close()

    best_solution = tools.selBest(population, 1)[0]
    #total_distance, transformed_polygons , Wires , Wire_labels = Transform_and_Rotate_Shapes(best_solution,MultiPoly_Object_List,global_connections, global_conn_place,Fixed_Polygons)
    transformed_polygons,Transform_and_Rotate_dict = Transform_and_Rotate_Shapes(best_solution,MultiPoly_Object_List,global_connections,global_conn_place,Fixed_Polygons)
    total_distance, Wires, _ = Find_wire_distance(best_solution,MultiPoly_Object_List,global_connections,global_conn_place,Fixed_Polygons,Transform_and_Rotate_dict)
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']
    fig, ax2 = plt.subplots()
    for i, multi_polygon in enumerate(transformed_polygons):
        color = colors[i % len(colors)]  # Cycle through the list of colors
        for polygon in multi_polygon.geoms:
            # Exterior coordinates split into x's and y's
            x, y = polygon.exterior.xy
            ax2.fill(x, y, alpha=0.5, color=color)

    for i,[point1, point2] in enumerate(Wires):
        line = geom.LineString([point1, point2])
        x, y = line.xy
        ax2.plot(x, y, linestyle='-', marker='o', color='grey')
    Title = 'Final Positions & Dist ' + str(total_distance)
    ax2.set_title(Title)
    #plt.axis('equal')
    File_Name = brd_path+'ShapelyGA.png'
    plt.savefig(File_Name)
    #plt.show()
    print('GA time ', time.time() - t1)
    #print('best_solution', best_solution)
    return(best_solution, transformed_polygons)

if __name__ == "__main__":
    brd_path = r'C:\Manav_Projects\Python_Programs\Design_Databases\S32k344_Simple_design/'
    GAstage_File = brd_path +'GAShapely.pickle'
    global_margins = [2,2]
    [MultiPoly_Object_List, global_connections, global_conn_place, _] = pickle.load(open(GAstage_File, 'rb'))
    GA_shapely_placement_analysis_stage3(brd_path, MultiPoly_Object_List, global_connections, global_conn_place, global_margins)

