import numpy

def select_parents(pop, fitness, num_parents):
    parents = numpy.empty((num_parents, pop.shape[1], pop.shape[2]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :, :] = pop[max_fitness_idx, :, :]
        fitness[max_fitness_idx] = -float("Inf")
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    crossover_point = numpy.uint8(offspring_size[1] / 2)

    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, 0:crossover_point, :] = parents[parent1_idx, 0:crossover_point, :]
        offspring[k, crossover_point:, :] = parents[parent2_idx, crossover_point:, :]
    return offspring

def mutation(offspring_crossover, num_mutations):
    for idx in range(offspring_crossover.shape[0]):
        for mutation_num in range(num_mutations):
            gene_idx = numpy.random.randint(low=0, high=8)
            offspring_crossover[idx, gene_idx, :] = numpy.random.randint(0, 8, size=2)
    return offspring_crossover
