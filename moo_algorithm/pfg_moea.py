import numpy as np
import random
from copy import deepcopy
import multiprocessing
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from moo_algorithm.metric import cal_hv_front
from population import Population
from graph.graph import Graph

def cal_knee_point(pop):
    knee_point = np.zeros(len(pop.indivs[0].objectives))
    m = len(pop.indivs[0].objectives)
    for i in range(m):
        knee_point[i] = 1e9
    for indi in pop.indivs:
        for i in range(m):
            knee_point[i] = min(knee_point[i], indi.objectives[i])
    return knee_point


def cal_nadir_point(pop):
    m = len(pop.indivs[0].objectives)
    nadir_point = np.zeros(m)
    for i in range(m):
        nadir_point[i] = -1e9
    for indi in pop.indivs:
        for i in range(m):
            nadir_point[i] = max(nadir_point[i], indi.objectives[i])

    SP = random.sample(pop.indivs, int(len(pop.indivs)/3))
    NDS_SP = NonDominatedSort(SP)
    for indi in NDS_SP: 
        for i in range(m):
            nadir_point[i] = max(nadir_point[i], indi.objectives[i])
    return nadir_point

def NonDominatedSort(SP):
    SP_copy = deepcopy(SP)
    ParetoFront = []
    for individual in SP_copy:
        individual.domination_count = 0
        individual.dominated_solutions = []
        for other_individual in SP_copy:
            if individual.dominates(other_individual):
                individual.dominated_solutions.append(other_individual)
            elif other_individual.dominates(individual):
                individual.domination_count +=1
        if individual.domination_count ==0:
            individual.rank = 0
            ParetoFront.append(individual)
    return ParetoFront

def Generation_PFG(pop, GK, knee_point, nadir_point, sigma):
    
    d = [(nadir_point[j] - knee_point[j] + 2*sigma) / GK for j in range(len(knee_point))]
    Grid = []
    for indi in pop.indivs:
        grid_indi = [(indi.objectives[j] - knee_point[j] + sigma) // d[j] for j in range(len(knee_point))]
        Grid.append(grid_indi)
    PFG = [[[] for _ in range(GK)] for _ in range(len(knee_point))]

    for idx, indi in enumerate(pop.indivs):
        for j in range(3):
            grid_value = int(Grid[idx][j])
            if 0 <= grid_value < GK:
                PFG[j][grid_value].append(indi)
    return PFG


class PFGMOEAPopulation(Population):
    def __init__(self, pop_size):
        super().__init__(pop_size)
        self.ParetoFront = []
    

    def fast_nondominated_sort_crowding_distance(self, indi_list):
        ParetoFront = [[]]
        for individual in indi_list:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in indi_list:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
                    # break
            if individual.domination_count == 0:
                individual.rank = 0
                ParetoFront[0].append(individual)
        i = 0
        while len(ParetoFront[i]) > 0:
            temp = []
            for individual in ParetoFront[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            ParetoFront.append(temp)
        for front in ParetoFront:
            self.calculate_crowding_distance(front)
        return ParetoFront

    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].objectives)):
                front.sort(key=lambda individual: individual.objectives[m])
                front[0].crowding_distance = 10 ** 9
                front[solutions_num - 1].crowding_distance = 10 ** 9
                m_values = [individual.objectives[m] for individual in front]
                scale = max(m_values) - min(m_values)
                if scale == 0: scale = 1
                for i in range(1, solutions_num - 1):
                    front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / scale

    # Crowding Operator
    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
                ((individual.rank == other_individual.rank) and (
                        individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def natural_selection(self):
        self.ParetoFront = self.fast_nondominated_sort_crowding_distance(self.indivs)
        new_indivs = []
        new_fronts = []
        front_num = 0
        while len(new_indivs) + len(self.ParetoFront[front_num]) <= self.pop_size:
            new_indivs.extend(self.ParetoFront[front_num])
            new_fronts.append(self.ParetoFront[front_num])
            if len(new_indivs) == self.pop_size:
                break
            front_num += 1
        self.calculate_crowding_distance(self.ParetoFront[front_num])
        self.ParetoFront[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
        number_remain = self.pop_size - len(new_indivs)
        new_indivs.extend(self.ParetoFront[front_num][0:number_remain])
        new_fronts.append(self.ParetoFront[front_num][0:number_remain])
        self.ParetoFront = new_fronts
        self.indivs = new_indivs

def filter_external(pareto):
    objectives = set()
    new_external_pop = []
    for indi in pareto:
        if tuple(indi.objectives) not in objectives:
            new_external_pop.append(indi)
            objectives.add(tuple(indi.objectives))
    return new_external_pop


def run_pfgmoea(processing_number, problem, indi_list, pop_size, max_gen, GK, sigma, crossover_operator, mutation_operator, 
            crossover_rate, mutation_rate, cal_fitness):
    pop = PFGMOEAPopulation(pop_size)
    pop.pre_indi_gen(indi_list)

    pool = multiprocessing.Pool(processing_number)
    arg = []
    for individual in pop.indivs:
        arg.append((problem, individual))
    result = pool.starmap(cal_fitness, arg)
    for individual, fitness in zip(pop.indivs, result):
        individual.objectives = fitness

    pop.natural_selection()
    # print("Generation 0: ", cal_hv_front(pop.ParetoFront[0], np.array([1, 1, 1])))

    history = {}
    # history[0] = [calculate_fitness(problem, i) for i in pop.ParetoFront[0]]

    for gen in range(max_gen+1):
        knee_point = cal_knee_point(pop)
        nadir_point = cal_nadir_point(pop)
        PFG = Generation_PFG(pop, GK, knee_point, nadir_point, sigma)
        offspring = []

        for j in range(len(knee_point)):
            for i in range(GK - 1):
                if len(PFG[j][i]) > 1:
                    for indi in PFG[j][i]:
                        if (random.random() < crossover_rate) and (len(PFG[j][i+ 1]) >1):
                            other_indi = random.choice(PFG[j][i + 1])
                            off1, off2 = crossover_operator(problem, indi, other_indi)
                            if random.random() < mutation_rate:
                                off1 = mutation_operator(problem, off1)
                                off2 = mutation_operator(problem, off2)
                            offspring.append(off1)
                            offspring.append(off2)

        arg = []
        for individual in offspring:
            arg.append((problem, individual))
        result = pool.starmap(cal_fitness, arg)
        for individual, fitness in zip(offspring, result):
            individual.objectives = fitness
        pop.indivs.extend(offspring)
        pop.natural_selection()

        # print("Generation {}: ".format(gen + 1), cal_hv_front(pop.ParetoFront[0], np.array([20000, 2000, 2000, 2000]))/20000/2000/2000/2000)

        pop.ParetoFront[0] = filter_external(pop.ParetoFront[0])

        history[gen] = [cal_fitness(problem, i) for i in pop.ParetoFront[0]]

    pool.close()

    # return pop.ParetoFront[0]

    # result = []
    # for each in pop.ParetoFront[0]:
    #     result.append(each.objectives)
    #     print(each.objectives)
    # return result

    # print(history)
    return history

if __name__ == "__main__":
    from utils import crossover_operator_lerk, mutation_operator_lerk, calculate_fitness_lerk, create_individual_pickup_lerk
    filepath = '.\\data\\dpdptw\\200\\LC1_2_1.csv'
    # filepath = '.\\data\\dpdptw\\400\\LC1_4_1.csv'
    graph = Graph(filepath)
    indi_list = [create_individual_pickup_lerk(graph) for _ in range(100)]
    result = run_pfgmoea(4, graph, indi_list, 100, 100, 5, 0.01, crossover_operator_lerk, mutation_operator_lerk, 0.9, 0.1, calculate_fitness_lerk)
    print(result)


# if __name__ == "__main__":
#     from LERK_utils import crossover_LERK, mutation_LERK, calculate_fitness_LERK, create_individual_LERK
#     filepath = '.\\data\\dpdptw\\200\\LC1_2_1.csv'
#     # filepath = '.\\data\\dpdptw\\400\\LC1_4_1.csv'
#     graph = Graph(filepath)
#     indi_list = [create_individual_LERK(graph) for _ in range(100)]
#     result = run_pfgmoea(4, graph, indi_list, 100, 100, 5, 0.01, crossover_LERK, mutation_LERK, 0.9, 0.1, calculate_fitness_LERK)
#     print(result)
