import numpy
N = 8

def select_parents(pop, fitness, num_parents):
    parents = []
    fitness_copy = fitness.copy()
    for _ in range(num_parents):
        max_fitness_idx = numpy.where(fitness_copy == numpy.max(fitness_copy))
        max_fitness_idx = max_fitness_idx[0][0]
        parents.append(pop[max_fitness_idx])
        fitness_copy[max_fitness_idx] = -float("Inf")
    return parents

def crossover(parents, offspring_size):
    offspring = []
    crossover_point = offspring_size[1] // 2

    for k in range(offspring_size[0]):
        parent1_idx = k % len(parents)
        parent2_idx = (k + 1) % len(parents)
        offspring_1 = parents[parent1_idx][:crossover_point] + parents[parent2_idx][crossover_point:]
        offspring.append(offspring_1)
    return offspring

def mutation(offspring_crossover, num_mutations):
    for idx in range(offspring_crossover.shape[0]):
        for mutation_num in range(num_mutations):
            gene_idx = numpy.random.randint(low=0, high=N)  # Flip a queen's position
            new_value = numpy.random.randint(0, N)
            offspring_crossover[idx, gene_idx] = new_value
    return offspring_crossover
