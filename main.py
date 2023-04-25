import numpy as numpy
import random

from utils import determine_best_solution, determine_worst_solution


def parse_data_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        num_items = int(lines[0].strip())
        items = []
        for line in lines[1:-1]:
            index, value, weight = map(int, line.strip().split())
            items.append((value, weight))
        max_weight = int(lines[-1].strip())
    return num_items, items, max_weight


def generate_starting_population(n, length):
    return numpy.random.randint(0, 2, (n, length))


def evaluation(population, weights, values, max_weight):
    fitness = []
    for element in population:
        total_value, total_weight = 0, 0
        for i, bit in enumerate(element):
            if bit:
                total_value += values[i]
                total_weight += weights[i]

        if total_weight > max_weight:
            total_value = 0

        fitness.append(total_value)
    return fitness


def selection(population, fitness, parents_number):
    sorted_indexes = numpy.argsort(fitness)[::-1]
    return population[sorted_indexes[:parents_number]]


def crossover(parents, children_number):
    children = numpy.empty(children_number)
    crossover_point = numpy.uint32(children_number[1] / 2)
    for k in range(children_number[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        children[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        children[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return children


def mutation(children, mutation_rate):
    for idx in range(children.shape[0]):
        if random.random() < mutation_rate:
            children[idx, random.randint(0, children.shape[1] - 1)] = 1 - children[
                idx, random.randint(0, children.shape[1] - 1)]
    return children


def knapsack_ea(weights, values, max_weight, n, length, num_generations, num_parents, mutation_rate):
    population = generate_starting_population(n, length)  # INITIALIZE population
    for generation in range(num_generations):
        fitness = evaluation(population, weights, values, max_weight)  # EVALUATE each candidate
        parents = selection(population, fitness, num_parents)  # SELECT parents
        children = crossover(parents, (n - parents.shape[0], length))  # RECOMBINE parents to create children
        children = mutation(children, mutation_rate)  # MUTATE children
        # Combine parents and offspring, and evaluate their fitness
        combined_population = numpy.vstack((population, children))
        combined_fitness = evaluation(combined_population, weights, values, max_weight)

        # Select the best individuals from the combined population to form the new population
        sorted_indices = numpy.argsort(combined_fitness)[::-1]
        population = combined_population[sorted_indices][:n]

    best_solution = population[numpy.argmax(fitness)]
    return best_solution, numpy.max(fitness)


def generate_knapsack_results_file():
    with open('results.txt', 'w') as file:
        for _ in range(10):
            best_solution, best_fitness = knapsack_ea(weights, values, max_weight, n, length, num_generations,
                                                      num_parents,
                                                      mutation_rate)
            print('Solution: ', best_solution)
            print('Fitness: ', best_fitness)

            best_solutions.append(best_solution)
            best_fitnesses.append(best_fitness)
        print(f'Best solution: {determine_best_solution(best_solutions, best_fitnesses)}', file=file)
        print(f'Worst solution: {determine_worst_solution(best_solutions, best_fitnesses)}', file=file)


if __name__ == '__main__':
    # weights = [10, 20, 30]
    # values = [60, 100, 120]
    # max_weight = 50
    # n = 100
    # length = len(weights)
    # num_generations = 100
    # num_parents = 20
    # mutation_rate = 0.2

    filename = 'rucsac-20.txt'
    items_number, items, max_weight = parse_data_file(filename)
    values, weights = zip(*items)

    n = 100
    length = len(weights)
    num_generations = 100
    num_parents = 80
    mutation_rate = 0.8

    best_solutions = []
    best_fitnesses = []

    generate_knapsack_results_file()
