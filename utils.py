import numpy


def determine_best_solution(best_solutions, best_fitnesses):
    best_fitness = numpy.max(best_fitnesses)
    best_solution = best_solutions[numpy.argmax(best_fitnesses)]
    return best_solution, best_fitness


def determine_worst_solution(best_solutions, best_fitnesses):
    worst_fitness = numpy.min(best_fitnesses)
    worst_solution = best_solutions[numpy.argmin(best_fitnesses)]
    return worst_solution, worst_fitness
